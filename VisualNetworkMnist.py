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
import cv2
from model import Net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netmodel = Net()

state_dict = torch.load("mnistcnn.pth.tar", map_location='cpu')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k
    if k[0:7] == "module.":
        name = k[7:]
    new_state_dict[name] = v
netmodel.load_state_dict(new_state_dict)
netmodel.eval()


def visualmodle(initimagefile,netmodel,layer,channel):

    class Suggest(nn.Module):
        def __init__(self, initdata=None):
            super(Suggest, self).__init__()

            # self.weight = nn.Parameter(torch.randn((1,1,28,28)))

            if initdata is not None:
                self.weight = nn.Parameter(initdata)
            else:
                data = np.random.uniform(-1, 1, (1, 1, 28, 28))
                data = data.astype("float32")
                data = torch.from_numpy(data)
                self.weight = nn.Parameter(data)

        def forward(self, x):
            x = x * self.weight
            return F.upsample(x, (28, 28), mode='bilinear', align_corners=True)

    netmodel.eval()

    if initimagefile is None:
        model = Suggest(None)
    else:
        img = cv2.imread(initimagefile)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = img.astype("float32")
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img / 256
        img = normalize(img)
        model = Suggest(img)

    optimizer = optim.SGD(model.parameters(), lr= 1.0)
    model.train()
    data = np.ones((1,1,28,28), dtype="float32")
    data =  torch.from_numpy(data)

    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()
        netmodel = netmodel.cuda()
        data = data.cuda()

    for i in range(100):
        output = model(data)

        netout=[]
        netint=[]
        def getnet(self, input, output):
            netout.append(output)
            netint.append(input)
        # print(netmodel.features)



######################################################
###############################
        # handle = netmodel.fc1.register_forward_hook(getnet)
        # output = netmodel(output)
        # output = netout[0][0,channel]


        handle = netmodel.conv1.register_forward_hook(getnet)
        output = netmodel(output)
        output = netout[0][0, channel, : , :]

        output = output.view(1,1, output.shape[0], output.shape[1])
        output = F.max_pool2d(output, netmodel.conv1.kernel_size[0])


###############################
######################################################

        netout=[]
        netint=[]

        # output = output.mean()
        target = output+256.0
        target = target.detach()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Inter:'+str(i) + "  loss:"+str(loss.cpu().detach().numpy()))
        handle.remove()

    # model = model.cpu()
    # netmodel = netmodel.cpu()
    # data = data.cpu()

    model.eval()
    output = model(data)
    out = output.view(28,28)
    out = out.cpu().detach().numpy()
    outmax = out.max()
    outmin = out.min()
    out = out * (256.0/(outmax-outmin)) - outmin * (256.0/(outmax-outmin))
    return out


# 128  7
# 512  30
for i in range(512):
    out = visualmodle(None,netmodel,1,i)
    cv2.imwrite("./imageout/L7_C"+str(i)+".jpg",out)

i=0