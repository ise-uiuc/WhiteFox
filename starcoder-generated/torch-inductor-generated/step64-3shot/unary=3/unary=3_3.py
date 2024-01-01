
import torch.nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(192, 10, 3, stride=1, padding=1, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(10, 60, 1, stride=1, padding=0)        
        self.conv3 = torch.nn.Conv2d(60, 44, 1, stride=1, padding=0)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(44, 13, 3, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = F.max_pool2d(v3, 2, 2, 0)
        v5 = v4 * v4
        v6 = v4 + v5
        v7 = self.conv3(v6)
        v8 = self.sigmoid2(v7)
        v9 = self.conv4(v8)
        v10 = self.sigmoid(v9)
        v11 = v10 + 1
        v12 = torch.relu(v11)
        v13 = torch.tanh(v12)
        return torch.max(v13)
# Inputs to the model
x1 = torch.randn(1, 192, 45, 84)
