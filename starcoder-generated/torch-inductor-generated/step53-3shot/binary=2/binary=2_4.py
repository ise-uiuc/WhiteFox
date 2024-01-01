
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, 17)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(x.len(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Input to the model
x = torch.randn(10, 3, 224, 224)
