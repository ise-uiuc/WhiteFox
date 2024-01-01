
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 3, stride=2, padding=1)
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv2(v2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv2(v2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv2(v2)
        v4 = v3 - torch.empty([0])
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
