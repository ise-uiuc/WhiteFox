
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=0)
 
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = torch.addmm(v3, v2, v1 * 8)
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
