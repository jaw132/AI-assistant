'''
This contains the model VGG-6 which is based on the famous VGG-11
for the task at hand this network proved too big and took too long to
train. Early signs were promising but decided on a much smaller model.
'''

import torch
from torch import nn
import torch.nn.functional as F

# define model parameters
image_size=96
image_shape = (3, image_size, image_size)

conv_dimensions = {'layer1': 64,
                   'layer2': 128,
                   'layer3': 256,
                   'layer4': 256}

# architecture based on VGG-11 but is much smaller
class VGG6(nn.Module):
  def __init__(self, input_dims, conv_dims):
    super(VGG6, self).__init__()
    self.convdim1, self.convdim2 = conv_dims['layer1'], conv_dims['layer2']
    self.convdim3, self.convdim4 = conv_dims['layer3'], conv_dims['layer4']
    self.input_channels = input_dims[0]

    self.conv1 = nn.Conv2d(self.input_channels, self.convdim1, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(self.convdim1, self.convdim2, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(self.convdim2, self.convdim3, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(self.convdim3, self.convdim4, kernel_size=3, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(self.convdim1)
    self.batchnorm2 = nn.BatchNorm2d(self.convdim2)
    self.batchnorm3 = nn.BatchNorm2d(self.convdim4) # only three batch norm layers as two conv layers are back to back
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(36864, 1000)
    self.fc2 = nn.Linear(1000, 1)
    self.dropout = nn.Dropout(0.2)

  def forward(self, X):
    X = self.maxpool1(self.batchnorm1(self.conv1(X)))
    X = self.maxpool2(self.batchnorm2(self.conv2(X)))
    X = self.maxpool3(self.batchnorm3(self.conv4(self.conv3(X))))

    X = X.view(-1, self.num_flat_features(X))
    X = F.relu(self.dropout(self.fc1(X)))
    X = torch.sigmoid(self.fc2(X))

    return X

  # returns number of flattened features in tensor
  def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
