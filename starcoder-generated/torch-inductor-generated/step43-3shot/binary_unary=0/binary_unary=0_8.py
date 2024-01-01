
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv1 = torch.nn.Conv2d(24, 40, 55, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(40, 24, 55, stride=1, padding=1)
        self.model2 = Model2()
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.gelu(v1)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        v5 = (v4 > v5).float()
        v6 = self.conv2(v3)
        v7 = self.relu(v6)
        v8 = v4 * v7
        v9 = self.model2(x2, x3, x4, x5)
        return v8 + v9
# Inputs to the model
import torch.nn.functional as F
x1 = torch.randn(16, 24, 15, 15)
x2 = torch.randn(16, 24, 15, 15)
x3 = torch.randn(16, 24, 15, 15)
x4 = torch.randn(16, 24, 15, 15)
x5 = torch.randn(16, 24, 15, 15)
