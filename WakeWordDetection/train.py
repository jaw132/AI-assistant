# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# define model architecture - todo remove hard coded numbers
class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.fc1 = nn.Linear(5511, 1375)
        self.batch1 = nn.BatchNorm1d(101)
        self.batch2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(101, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.batch1((self.fc1(x))))
        x = torch.reshape(x, (batch_size, 1375, 101))
        x, idx = self.lstm(x)
        x = torch.reshape(x, (batch_size, 128, 1375))
        x = self.batch2(x)
        x = self.dropout(x)
        x = torch.reshape(x, (batch_size, 1375, 128))
        x = self.sigmoid(self.fc2(x))
        return x

# instantiate model and make sure it was float parameters
net_v2 = Net_v2()
net_v2 = net_v2.float()

#parameter: batch_size=64, lr=0.01, optmizier = adamW, custom loss

'''
Some other architectures that didn't work so well
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv1d(in_channels=101, out_channels=196, kernel_size=15, stride=4)
        self.batch1 = nn.BatchNorm1d(196)
        self.dropout = nn.Dropout(p=0.2)
        self.gru1 = nn.GRU(196, 128, batch_first=True, dropout=0.2)
        self.batch2 = nn.BatchNorm1d(128)
        self.gru2 = nn.GRU(128, 128, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.batch1((self.conv(x))))
        x = self.dropout(x)
        x = torch.reshape(x, (batch_size, 1375, 196))
        x, idx = self.gru1(x)
        x = torch.reshape(x, (batch_size, 128, 1375))
        x = self.batch2(x)
        x = torch.reshape(x, (batch_size, 1375, 128))
        x, idx = self.gru2(x)
        x = torch.reshape(x, (batch_size, 128, 1375))
        x = self.batch2(x)
        x = self.dropout(x)
        x = torch.reshape(x, (batch_size, 1375, 128))
        x = self.sigmoid(self.fc(x))
        return x


class Net_v3(nn.Module):
    def __init__(self):
        super(Net_v3, self).__init__()
        self.conv = nn.Conv1d(in_channels=101, out_channels=128, kernel_size=15, stride=4)
        self.batch = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.batch((self.conv(x))))
        x = torch.reshape(x, (batch_size, 1375, 128))
        x, idx = self.lstm(x)
        x = torch.reshape(x, (batch_size, 128, 1375))
        x = self.batch(x)
        x = self.dropout(x)
        x = torch.reshape(x, (batch_size, 1375, 128))
        x = self.sigmoid(self.fc(x))
        return x
'''

# set hyperparameters
learning_rate=0.01
momentum=0.9
epochs = 30
batch_size = 64
model = net_v2
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
load_data_params = {'batch_size': 64,
                    'shuffle': True}

# pytorch's dataloader requires data to be in a Dataset object,
# which needs len and getitem functions
class Dataset(torch.utils.data.Dataset):
    def __init__(self, audio, labels):
        self.audio = audio
        self.labels = labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        return self.audio[index], self.labels[index]


''' a major problem with this dataset is that most of the data is labelled 0 so a model
 can perform 'well' by some metrics by always predicting 0, we don't want this
 create a custom loss function that looks punishes mislabelling a '1' more to even out
 the imbalance'''
def custom_loss(criterion, predictions, labels, lambda_pct=0.5):
  #find ones in labels
  number_samps = predictions.shape[0]

  one_indexes = np.where(labels == 1)
  one_preds = predictions[one_indexes]

  zero_indexes = np.where(labels == 0)
  zero_preds = predictions[zero_indexes]

  zero_loss = criterion(zero_preds, torch.zeros_like(zero_preds))
  one_loss = criterion(one_preds, torch.ones_like(one_preds))
  loss = lambda_pct*zero_loss+(1-lambda_pct)*one_loss

  return loss



def train_model(model, epochs, audio, labels):

    training_set = Dataset(audio, labels)
    training_generator = torch.utils.data.DataLoader(training_set, **load_data_params)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch: ' + str(epoch + 1))

        for inputs, labels in training_generator:
            inputs, labels = inputs.float(), labels.float()
            batch_size = len(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = torch.reshape(outputs, (batch_size, 1, 1375))
            loss = custom_loss(criterion, outputs, labels, 0.5)
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print("Loss: " + str(loss.item()))

    print('Finished Training')

def main():
    X_tensor, Y_tensor = torch.load("train_audio"), torch.load("train_labels")
    train_model(model, epochs, X_tensor, Y_tensor)
    torch.save(net_v2, "trigger_word_model")

if __name__ == "__main__":
    main()