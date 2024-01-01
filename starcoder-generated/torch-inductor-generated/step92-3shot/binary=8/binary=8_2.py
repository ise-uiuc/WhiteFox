
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.dp1 = nn.Dropout2d(0.25)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dp2 = nn.Dropout2d(0.25)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dp3 = nn.Dropout2d(0.25)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dp4 = nn.Dropout2d(0.25)
        self.dp5 = nn.Dropout2d(0.25)
        
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.dp1(h)
        h = self.bn1(h)
        h = F.relu(self.conv2(h))
        h = self.dp2(h)
        h = self.bn2(h)
        h = F.relu(self.conv3(h))
        h = self.dp3(h)
        h = self.bn3(h)
        conv4 = self.conv4(h)
        bn4 = self.bn4(conv4)
        dp4 = self.dp4(bn4)
        dp5 = self.dp5(dp4)
        return dp5

# Inputs to the model
x = torch.randn(64, 1, 28, 28)
