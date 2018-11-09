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



# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1,  8, kernel_size=5)
#         self.conv2 = nn.Conv2d(8,  8, kernel_size=3)
#         self.conv3 = nn.Conv2d(8, 2, kernel_size=5)
#
#     def forward(self, x):
#
#
#         x = F.sigmoid(F.max_pool2d(self.conv1(x), 2))
#         x = F.sigmoid(F.max_pool2d(self.conv2(x), 2))
#         x = self.conv3(x)
#
#         x = x.view(-1, 1*2)
#
#         return F.log_softmax(x, dim=1)

#####  Master
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,  2, kernel_size=5 , stride=2)
        # self.conv2 = nn.Conv2d(2,  2, kernel_size=3)
        # self.conv3 = nn.Conv2d(2, 2, kernel_size=5)
        self.fc1 = nn.Linear(12 * 12 * 2, 2)

    def forward(self, x):


        x = self.conv1(x)
        x = x.view(-1,12*12 *2)
        x = self.fc1(x)



        # x = F.sigmoid(F.max_pool2d(self.conv1(x), 2))
        # x = F.sigmoid(F.max_pool2d(self.conv2(x), 2))
        # x = self.conv3(x)
        #
        # x = x.view(-1, 1*2)

        return F.log_softmax(x, dim=1)