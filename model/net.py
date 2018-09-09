import torch.nn as nn
import torch.nn.functional as F
import torch

## Define network
class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred

class ThreeLayerNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        y_pred = self.linear3(h2_relu)
        return y_pred

def loss_fn(outputs, targets):
    return -torch.sum((outputs - targets)**2)

