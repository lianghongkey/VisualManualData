from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from model import Net
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random



batchsize = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: "+ str(device))


class ManualDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, index):

        # data = np.random.random_integers(0,255,(28,28))
        data = np.zeros((28,28),dtype="float32")
        data = data.astype("float32")

        # basex = random.randint(3, 25)
        # basey = random.randint(3, 25)

        num1 = random.random()
        num2 = random.random()

        if num1>0.5 and num2>0.5:
            data[14,14]=num1
            data[14,15]=num2
            label = 1
        else:
            # if num1>0.5:
            #     data[14,14] = 1
            # else:
            #     data[14,14] = 0
            #
            # if num2>0.5:
            #     data[14,15] = 1
            # else:
            #     data[14,15] = 0
            data[14, 14] = num1
            data[14, 15] = num2
            label = 0

        data = (data-128)/256.0
        img = torch.from_numpy(data)
        img = img.view(1,28,28)
        return img,label

    def __len__(self):
        return 10000

train_loader = torch.utils.data.DataLoader(
        ManualDataset(transform=transforms.Compose([
                           # transforms.ColorJitter(0.2,0.2),
                           # transforms.RandomRotation(30),
                           # transforms.RandomResizedCrop(28),
                           transforms.ToTensor(),
                           transforms.Normalize((128,), (256,)),])),
                       batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
        ManualDataset(transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((128,), (256,))])),
        batch_size=batchsize, shuffle=True, num_workers=2)


model = (Net()).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 930 == 0 and batch_idx>0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

for epoch in range(1, 20):
    train(epoch)
    test()

torch.save(model.state_dict(),"mnistcnn.pth.tar")

