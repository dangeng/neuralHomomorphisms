'''
Learn a function g and g^{-1} such that g^{-1}(g(x)g(y)) = x+y
Should learn to map to an exponential
'''

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

domain = 2
learning_rate = .001
batch_size = 16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.act = nn.ReLU()
        self.mlp = nn.Sequential(self.fc1, self.act, 
                            self.fc2, self.act, 
                            self.fc3, self.act, 
                            self.fc4, self.act, 
                            self.fc5)

    def forward(self, x):
        return self.mlp(x)

g = Net()
g_inv = Net()
params = list(g.parameters()) + list(g_inv.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
criterion = nn.MSELoss()

# Train on g^{-1}(g(x)g(y)) = x + y

for i in range(1000):
    optimizer.zero_grad()

    # Generate data
    x = np.random.uniform(-domain, domain, (batch_size, 1))
    y = np.random.uniform(-domain, domain, (batch_size, 1))
    x, y = torch.tensor(x).float(), torch.tensor(y).float()

    # Calculate cycle
    gx, gy = g(x), g(y)
    out = g_inv(gx*gy)
    loss = criterion(out, x+y)

    # Optimize
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss.item())

def get_div(y, n=1):
    return y[n:] / y[:-n]

x = np.linspace(-domain*2, domain*2, 100)
y = g(torch.tensor(x).float().unsqueeze(1))
y = y.detach().numpy()[:, 0]

plt.plot(x, y)
plt.show()

div = get_div(y, n=10)
plt.plot(x[:-10], div)
plt.show()
