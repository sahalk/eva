# -*- coding: utf-8 -*-
"""test_model.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U8j6oTPH54LIfuExjtLcStn_AzEOMCT1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # Convolution Block 1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    # Max Pooling 1
    self.pool1 = nn.Sequential(
        nn.MaxPool2d(2,2)
    )

    # Convolution Block 2
    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )

    # Convolution Block 3
    self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding = (2,2), bias=False, dilation=(2,2)),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    # Max Pooling 2
    self.pool2 = nn.Sequential(
        nn.MaxPool2d(2,2)
    )

    # Convolution Block 4
    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    # Convolution Block 5
    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1,1), groups=256, bias=False),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    # Max Pooling 3
    self.pool3 = nn.Sequential(
        nn.MaxPool2d(2,2)
    )

    # Convolution Block 6
    self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.convblock7 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), groups=256, bias=False),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
    )

    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=(0, 0), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=5)
    )
    self.convblock9 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=(0, 0), bias=False),
    )
    
  def forward(self, x):
      x = self.convblock1(x)
      x = self.pool1(x)
      x = self.convblock2(x)
      x = self.convblock3(x)
      x = self.pool2(x)
      x = self.convblock4(x)
      x = self.convblock5(x)
      x = self.pool3(x)
      x = self.convblock6(x)
      x = self.convblock7(x)
      x = self.convblock8(x)
      x = self.gap(x)
      x = self.convblock9(x)

      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)

class Loader(object):
  def __init__(self, data_mean, data_std_dev):
    super(Loader, self).__init__()
    self.cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.cuda else "cpu")

    self.model = Net().to(self.device)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std_dev)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    self.classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def disp_image(self, trainloader):
    def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(self.trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

  def print_sum(self):
    summary(self.model, input_size=(3, 32, 32))

  def train(self, limit, learning_rate=0.01, momentum=0.9):
    self.train_losses = []
    self.test_losses = []
    self.train_acc = []
    self.test_acc = []
    optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

    for i in range(limit):
        self.model.train()
        pbar = tqdm(self.trainloader)
        correct = 0
        processed = 0

        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, labels)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            self.test()
      

  def test(self):
      self.model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, labels in self.testloader:
              data, labels = data.to(self.device), labels.to(self.device)
              output = self.model(data)
              test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(labels.view_as(pred)).sum().item()

      self.test_loss /= len(self.testloader.dataset)
      self.test_losses.append(test_loss)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(self.testloader.dataset),
          100. * correct / len(self.testloader.dataset)))
      
      self.test_acc.append(100. * correct / len(self.testloader.dataset))
    
  def class_accuracy(self):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in self.testloader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            self.classes[i], 100 * class_correct[i] / class_total[i]))