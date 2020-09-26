import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.01

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # Input Layer
    self.input = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
    )

    # Convolution Block 1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
    )

    # Convolution Block 2
    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Dropout(dropout_value)
    )

    # Max Pooling 1
    self.pool1 = nn.Sequential(
        nn.MaxPool2d(2,2)
    )

    # Convolution Block 3
    self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Dropout(dropout_value)
    )

    # Convolution Block 4
    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Dropout(dropout_value)
    )

    # Convolution Block 5
    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding = (1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Dropout(dropout_value)
    )

    # Max Pooling 2
    self.pool2 = nn.Sequential(
        nn.MaxPool2d(2,2)
    )

    # Convolution Block 6
    self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=(1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Dropout(dropout_value)
    )

    # Convolution Block 7
    self.convblock7 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1,1), groups=256, bias=False),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    # Convolution Block 8
    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    # GAP Layer
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=5)
    )

    # Output Layer
    self.convblock9 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=(0, 0), bias=False),
    )


  def forward(self, x):
      x = self.input(x)
      x = self.convblock1(x)
      x = self.convblock2(x)
      x = self.pool1(x)
      x = self.convblock3(x)
      x = self.convblock4(x)
      x = self.convblock5(x)
      x = self.pool2(x)
      x = self.convblock6(x)
      x = self.convblock7(x)
      x = self.convblock8(x)
      x = self.gap(x)
      x = self.convblock9(x)

      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)