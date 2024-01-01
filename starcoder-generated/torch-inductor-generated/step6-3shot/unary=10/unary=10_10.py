
import torch.nn as nn
import torch.nn.functional as F
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2= nn.Conv2d(3, 1, 5, padding=0)
        self.conv3 = nn.Conv2d(3, 3, 1, padding=2, bias=False)
 
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu6(self.conv2(x))
        x = F.relu6(self.conv3(x))
        return x

# Initializing the model
m = Model()
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2= nn.Conv2d(3, 1, 5, padding=0)
        self.conv3 = nn.Conv2d(3, 3, 1, padding=2, bias=False)
 
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu6(self.conv2(x))
        x = F.relu6(self.conv3(x))
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 56, 56)
