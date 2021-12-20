
import numpy as np
import pandas as pd

rankdata_000 = pd.read_csv(
    r'D:\Study\Python\Workspace\rainbowSixSiege_analysis\datadump_s5_ranked_data\datadump_s5-000.csv')
col = rankdata_000.columns

data = rankdata_000.loc[:,
       ['matchid', 'roundnumber', 'gamemode', 'mapname', 'roundduration', 'skillrank', 'role', 'haswon']]

opdata = pd.get_dummies(rankdata_000['operator'], drop_first=False, prefix='OP')
newdata = pd.merge(data, opdata, left_index=True, right_index=True)
new_matchid = newdata['matchid'].map(str) + "_" + newdata['roundnumber'].map(str) + "_" + newdata['role'].map(str)
newdata = newdata.drop(labels=['matchid', 'roundnumber', 'role'], axis=1)
newdata.insert(0, 'newmatchid', new_matchid, allow_duplicates=False)

newdata['roundduration'].describe()

newdata['gamemode'] = newdata['gamemode'].replace({'HOSTAGE': 1, 'BOMB': 2, 'SECURE_AREA': 3})
newdata['mapname'] = newdata['mapname'].replace(
    {'CLUB_HOUSE': 1, 'PLANE': 2, 'KANAL': 3, 'HEREFORD_BASE': 4, 'CONSULATE': 5,
     'YACHT': 6, 'OREGON': 7, 'BORDER': 8, 'SKYSCRAPER': 9, 'BANK': 10, 'COASTLINE': 11,
     'BARTLETT_U.': 12, 'HOUSE': 13, 'KAFE_DOSTOYEVSKY': 14, 'FAVELAS': 15, 'CHALET': 16})
newdata['skillrank'] = newdata['skillrank'].replace(
    {'Gold': 4, 'Unranked': 0, 'Platinum': 5, 'Silver': 3, 'Bronze': 1, 'Copper': 2})

print(newdata.shape)
# skillrank disappear
res = newdata.groupby(newdata['newmatchid']).agg({max}).reset_index()
print(res.shape)

SCALE = 100000
label = np.array(res.loc[:SCALE - 1, :]['haswon'])
print(label.shape)
train_data = np.array(
    res.drop(labels=['newmatchid', 'haswon', 'gamemode', 'mapname', 'roundduration'], axis=1).loc[:SCALE - 1, :])
print(train_data.shape)

# PyTorch MLP

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

learning_rate = 0.05
batch_size = 1000
epochs = 10

# prepare data
c = np.column_stack((train_data,label))  # 将y添加到x的最后一列
np.random.shuffle(c)
shuffled_data = c[:,:-1]  # 乱序后的x
shuffled_label = c[:,-1]  # 同等乱序后的y
SIZE = int(SCALE * 0.7)
train_x = shuffled_data[:SIZE]
test_x = shuffled_data[SIZE:]
y = []
for n in range(label.shape[0]):
    if label[n] == 0:
        y.append((1, 0))
    else:
        y.append((1, 0))
train_y = np.array(shuffled_label[:SIZE])
test_y = np.array(shuffled_label[SIZE:])

class CustomDataset(Dataset):
    def __init__(self, train):
        if train:
            self.data = train_x.astype('float32')
            self.labels = train_y
        else:
            self.data = test_x.astype('float32')
            self.labels = test_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.labels[idx]
        return d, l


train_dataset = CustomDataset(train=True)
test_dataset = CustomDataset(train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(35, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")