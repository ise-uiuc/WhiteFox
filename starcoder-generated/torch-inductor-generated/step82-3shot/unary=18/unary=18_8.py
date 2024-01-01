
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,1), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(65536, 43793)
        self.linear1 = nn.Linear(43793, 43793)
        self.linear2 = nn.Linear(43793, 43793)
        self.linear3 = nn.Linear(43793, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.flatten(v2)
        v4 = self.linear0(v3)
        v5 = self.linear1(v4)
        v6 = self.linear2(v5)
        v7 = math.sigmoid(self.linear3(v6))
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 524, 359)
