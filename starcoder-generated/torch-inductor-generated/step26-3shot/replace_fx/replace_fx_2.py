
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.RNN(
                32,
               16,
               2,
               batch_first=True,
               dropout=0.0,
               bidirectional=True)
    def forward(self, x):
        x4 = self.net(x)
        x5 = torch.rand_like(x4, requires_grad=True)
        return x5
# Inputs to the model
x = torch.randn(10, 13, 32)
