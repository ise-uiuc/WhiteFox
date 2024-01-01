
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(3, 3)
        self.maxpool = nn.MaxPool2d(4, stride=4)
        self.fc_2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.maxpool(x)
        x = self.fc_2(x)
        x = F.flatten(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3)
