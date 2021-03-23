import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

# the following is code to import kaggle datasets into google colab
# using the kaggle api
'''
! pip install -q kaggle
from google.colab import files
files.upload()
###upload the kaggle.json file ###
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download 'aliasgartaksali/human-and-non-human'
!unzip human-and-non-human.zip
'''


# function to visualise the data
def visualise_images(path):
    test_image = Image.open(path)
    plt.imshow(test_image)

# define the parameters for the images and model
train_path = 'human-and-non-human/train_set/train_set'
test_path = 'human-and-non-human/test_set/test_set'
image_size=96
image_shape = (3, image_size, image_size)
batch_size = 32
load_data_params = {'batch_size': batch_size,
                    'shuffle': True}

learning_rate=0.01
momentum=0.9
epochs = 5

class CNN(nn.Module):
  def __init__(self, input_dim):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.dropout = nn.Dropout(0.2)
    self.fc1 = nn.Linear(30976, 128)
    self.fc2 = nn.Linear(128, 1)


  def forward(self, X):
    X = self.maxpool1(F.relu(self.batchnorm1(self.conv1(X))))
    X = self.maxpool2(F.relu(self.batchnorm2(self.conv2(X))))

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


def train(model, image_loader, criterion, optimizer):
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch: ' + str(epoch + 1))

        epoch_loss, number_batches=0, 0
        for inputs, labels in image_loader:
            number_batches+=1
            inputs, labels = inputs.float(), labels.float()
            labels = labels.view(labels.shape[0], 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Loss: " + str(epoch_loss/number_batches))
    return model


def evaluate_model(model, image_loader):
    # check accuracy on whole test set
    total = 0
    total_samples = 0

    for test_input, test_labels in image_loader:
        test_input, test_labels = test_input.detach().float(), test_labels.float()
        test_labels = test_labels.view(test_labels.shape[0], 1)
        total_samples += len(test_labels)

        test_output = model(test_input)
        ones = torch.ones_like(test_output)
        zeroes = torch.zeros_like(test_output)
        processed_output = torch.where(test_output >= 0.5, ones, zeroes)

        for i in range(len(ones)):
            if test_labels[i] == processed_output[i]:
                total += 1

    total = total / total_samples
    return total


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((image_size, image_size)),  # ensure all images are same size
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # load in training data from folder seg_test
    trainset = torchvision.datasets.ImageFolder(root=train_path,
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, **load_data_params)
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(testset, **load_data_params)

    model = CNN(image_shape).float()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    model=train(model, trainloader, criterion, optimizer)
    accuracy=evaluate_model(model, testloader)
    print(accuracy)
    #accuracy was 99.4% on whole test set
    torch.save(model, 'human-or-not-model')


if __name__ == '__main__':
    main()