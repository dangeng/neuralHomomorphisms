'''
Learn to find a map g: \mathbb{R} \to F, where F is image to image functions
Such that g(x + y) = g(x)(g(y)) and all g(x) keeps the color histogram constant
In addition, train a g^{-1} in the sense that g^{-1}(I, f(I)) = x

We thus have the constraints g^{-1}(I, g(x)(g(y))(I)) = x + y and g(x)(g(y))(I) and I have the same color histogram

There is a very simple transform that satisfies this relation. That is g maps a real number \theta to a rotation by \theta.
'''

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from networks import define_G

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.fc = nn.Linear(7744, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(batch_size, -1)
        return self.fc(x)

domain = 2
learning_rate = .001
batch_size = 16

g = define_G(2, 1, 32, 'resnet_6blocks').cuda()
g_inv = Net().cuda()
params = list(g.parameters()) + list(g_inv.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
criterion = nn.MSELoss()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
	   transform=transforms.Compose([
	       transforms.ToTensor(),
	       transforms.Normalize((0.1307,), (0.3081,))
	   ])),
	batch_size=batch_size, shuffle=True)

def add_param_channel(x, im):
    x = np.tile(x[:,0], (28,28,1)).transpose(2,0,1)
    x = torch.tensor(x).float().unsqueeze(1).cuda()
    return torch.cat((x, im), dim=1)

for i, (I, target) in enumerate(train_loader):
    I, target = I.cuda(), target.cuda()

    x = np.random.uniform(-2, 2, (batch_size, 1))
    y = np.random.uniform(-2, 2, (batch_size, 1))

    # g_x(I)
    xI = add_param_channel(x, I)
    gxI = g(xI)

    # g_y(g_x(I))
    ygxI = add_param_channel(y, gxI)
    gygxI = g(ygxI)

    x = torch.tensor(x).float().cuda()
    y = torch.tensor(y).float().cuda()

    # g^{-1}(g_y(g_x(I))) should = x + y
    I_gygxI = torch.cat((I, gygxI), dim=1)
    inv = g_inv(I_gygxI)

    # optimize loss
    optimizer.zero_grad()
    loss = criterion(inv, x+y)
    loss.backward()
    print(loss.item())
    optimizer.step()
