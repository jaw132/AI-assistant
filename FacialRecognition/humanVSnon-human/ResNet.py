#import libraries
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torchvision
import torch.optim as optim

# initialise model parameters
filter_sizes = {
    'initial_conv': 64,
    'res_block1': [64, 64, 256],
    'res_block2': [128, 128, 512],
    'res_block3': [256, 256, 1028],
    'res_block4': [512, 512, 2048]
}

image_shape = (3, 224, 224)
classes = 6

# set hyperparameters
learning_rate = 0.01
momentum = 0.9
epochs = 10
batch_size = 16
num_workers = 2
load_data_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': num_workers}

'''
define a series of classes that will make up the building blocks of ResNet50
identity blocks - three convolutional layers with an identity skip connection
conv blocks - three convolutional layers with an convolutional skip connection
residual blocks - which are comprised of a conv block and then a number of identity blocks
'''

class identity_block(nn.Module):
    def __init__(self, input_dim, filters):
        super(identity_block, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, filters[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(filters[0])
        self.batch_norm2 = nn.BatchNorm2d(filters[1])
        self.batch_norm3 = nn.BatchNorm2d(filters[2])

    def forward(self, X):
        # make a copy of the input for the skip connnection later
        X_copy = X.clone()

        X = F.relu(self.batch_norm1(self.conv1(X)))
        X = F.relu(self.batch_norm2(self.conv2(X)))
        X = self.batch_norm3(self.conv3(X))
        X = X + X_copy
        X = F.relu(X)
        return X

class conv_block(nn.Module):
  def __init__(self, input_dim, filters, stride):
    super(conv_block, self).__init__()
    self.conv1 = nn.Conv2d(input_dim, filters[0], kernel_size=1, stride=stride)
    self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1)
    self.conv_shortcut = nn.Conv2d(input_dim, filters[2], kernel_size=1, stride=stride)
    self.batch_norm1 = nn.BatchNorm2d(filters[0])
    self.batch_norm2 = nn.BatchNorm2d(filters[1])
    self.batch_norm3 = nn.BatchNorm2d(filters[2])

  def forward(self, X):
    # convolutional residual jump
    X_copy = X.clone()
    X_copy = self.batch_norm3(self.conv_shortcut(X_copy))

    X = F.relu(self.batch_norm1(self.conv1(X)))
    X = F.relu(self.batch_norm2(self.conv2(X)))
    X = self.batch_norm3(self.conv3(X))
    X = X + X_copy
    X = F.relu(X)
    return X


class residual_block(nn.Module):
    def __init__(self, identity_layers, input_dim, filters, stride):
        '''
        inputs:
        identity_layers - number of identity_block (int)
        input_dim - number of channels of input tensor (int)
        filters - list of the sizes of filters for each conv layers within block

        outputs:
        residual_block made of one conv block and 'identity_layers' identity blocks
        '''
        super(residual_block, self).__init__()
        self.conv_block = conv_block(input_dim, filters, stride)
        self.identity_blocks = nn.ModuleList([identity_block(filters[-1], filters) for i in range(identity_layers)])

    def forward(self, X):
        X = self.conv_block(X)
        for id_block in self.identity_blocks:
            X = id_block(X)

        return X


'''
Building ResNet50
'''
class ResNet(nn.Module):
  def __init__(self, input_shape, classes, filters):
    '''
    inputs:
    input_shape - shape of the input images (1x3 tuple)
    classes - the number of classes for the model to classify (int)
    filters - dictionary containing all conv. filter dimensions
    '''
    super(ResNet, self).__init__()
    self.input_shape = input_shape
    self.classes = classes
    self.block_sizes = [2, 3, 5, 3]
    self.filters = filters
    if len(self.filters) != 5:
      print("ResNet50 requires 5 filter dimension inputs")

    ####################################################################
    # initial convolutional layer                                      #
    ####################################################################
    self.init_conv = nn.Conv2d(in_channels=self.input_shape[0], out_channels=self.filters['initial_conv'], kernel_size=7, stride=2)
    self.init_bn = nn.BatchNorm2d(num_features=self.filters['initial_conv'])
    self.init_maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    ####################################################################
    # first residual block                                             #
    ####################################################################

    # one conv block followed by two identity blocks
    self.residual_block1 = residual_block(self.block_sizes[0], self.filters['initial_conv'], self.filters['res_block1'], stride=1)

    ####################################################################
    # second residual block                                            #
    ####################################################################

    # one conv block followed by three identity blocks
    self.residual_block2 = residual_block(self.block_sizes[1], self.filters['res_block1'][-1], self.filters['res_block2'], stride=2)

    ####################################################################
    # third residual block                                             #
    ####################################################################

    # one conv block followed by five identity blocks
    self.residual_block3 = residual_block(self.block_sizes[2], self.filters['res_block2'][-1], self.filters['res_block3'], stride=2)

    ####################################################################
    # fourth residual block                                            #
    ####################################################################

    # one conv block followed by three identity blocks
    self.residual_block4 = residual_block(self.block_sizes[3], self.filters['res_block3'][-1], self.filters['res_block4'], stride=2)

    ####################################################################
    # final layers of model                                            #
    ####################################################################

    self.fin_avgpool = nn.AvgPool2d(kernel_size=7)
    self.fc = nn.Linear(self.filters['res_block4'][-1], self.classes)


  def forward(self, X):
    X = self.init_maxpool(F.relu(self.init_bn(self.init_conv(X))))

    X = self.residual_block1(X)
    X = self.residual_block2(X)
    X = self.residual_block3(X)
    X = self.residual_block4(X)

    X = self.fin_avgpool(X)
    X = X.view(-1, self.num_flat_features(X))
    X = self.fc(X)

    return X

  # returns number of flattened features in tensor
  def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(model, optimizer, criterion):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # load in training data from folder seg_test
    trainset = torchvision.datasets.ImageFolder(root='/seg_test',
                                                transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, **load_data_params)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch: ' + str(epoch + 1))

        for inputs, labels in trainloader:
            inputs, labels = inputs.float(), labels.float()

            print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print("Loss: " + str(loss.item()))


def main():
    model = ResNet(image_shape, classes, filter_sizes).float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    train(model, optimizer, criterion)


if __name__ == "__main__":
    main()