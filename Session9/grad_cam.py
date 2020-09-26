import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCam():
  def __init__(self, model, test_loader, target_layers, device):
    self.model = model
    self.test_loader = test_loader
    self.model.eval()
    self.target_layers = target_layers
    self.device = device
    self.name_layer_map =  dict(self.model.named_modules())
    classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  def get_overlays(self, required_class, one_hot_encoded_class=None):
    
    overlays = {}

    one_hot_encoded_class = one_hot_encoded_class if one_hot_encoded_class else required_class

    # Get labels
    required_label = classes.index(required_class)
    one_hot_label = classes.index(one_hot_encoded_class)

    # get some random training images for the corresponding label
    dataiter = iter(self.test_loader)
    while True:
      images, labels = dataiter.next()
      if required_label in labels:
        images = images[labels == required_label]
        labels = labels[labels == required_label]
        break
    
    # Collect gradients and activations
    handles = []
    activations = {}
    gradients = {}

    def get_activations(layer_name):
        def forward_hook(module, input, output):
            activations[layer_name] = output.detach()
        return forward_hook

    def get_gradients(layer_name):
        def backward_hook(module, grad_in, grad_out):
            gradients[layer_name] = grad_out[0].detach()
        return backward_hook

    # Register hooks to collect gradients and activations
    for layer_name in self.target_layers:
        if layer_name in self.name_layer_map:
            forward_handler = self.name_layer_map[layer_name].register_forward_hook(get_activations(layer_name))
            backward_handler = self.name_layer_map[layer_name].register_backward_hook(get_gradients(layer_name))
            handles.append(forward_handler)
            handles.append(backward_handler)

    # Collect a sample image and create a tensor
    x = images[0:1, :, :, :].to(self.device)
    x = x.requires_grad_(requires_grad=True)
    
    # Forward pass
    self.model.zero_grad()
    logits = self.model.logits(x)

    # Get one hot encoded vector for back propogation
    one_hot = torch.zeros_like(logits).to(self.device)
    ids = torch.LongTensor([[one_hot_label]]).to(self.device)
    one_hot = one_hot.scatter_(1, ids, 1.0)

    # Backward pass
    logits.backward(gradient=one_hot, retain_graph=True)

    # Get overlays for all layers
    for layer_name in self.target_layers:
      if layer_name in gradients and layer_name in activations:

        # GAP on gradients
        weights = F.adaptive_avg_pool2d(gradients[layer_name], 1)

        # Weight computation
        heat_map = torch.mul(activations[layer_name], weights).sum(dim=1, keepdim=True)

        # Resize, collect and color heat maps
        H, W = 32, 32
        heat_map = F.relu(heat_map)

        _, _, _H, _W = heat_map.shape
        heat_map = heat_map.view((_H, _W))
        _heat_map = heat_map.to('cpu').numpy()

        _heat_map = cv2.resize(_heat_map, (H, W))
        _heat_map = cv2.normalize(_heat_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _heat_map = cv2.applyColorMap(_heat_map, cv2.COLORMAP_JET)
        _heat_map = cv2.cvtColor(_heat_map, cv2.COLOR_RGB2BGR)

        image = images[0, :, :, :].permute(1, 2, 0).numpy()
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Create a weight average of the results
        overlay = cv2.addWeighted(_heat_map, 0.25, image, 0.75, 0.0)

        # Add to overlay computation
        overlays[layer_name] = overlay
  
    for handle in handles:
      handle.remove()
  
    return overlays

  def display_heatmaps(self, overlays):
    fig = plt.figure(figsize=(25, 8))
    fig.tight_layout()

    for i, (key, value) in enumerate(overlays.items()):
        ax = fig.add_subplot(1, len(overlays), i + 1)
        plt.imshow(value)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(key)

    plt.show()