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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5,bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3,bias=False)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=3,bias=False)

        self.fc1 = nn.Linear(9 * 64, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        # x = x.view(-1, 9 * 64)
        # x = self.fc1(x)

        x = F.max_pool2d(x,3)
        x = x.view(-1, 2)



        return F.log_softmax(x, dim=1)
