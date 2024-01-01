
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(3, 4, 8, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)*1
        v2 = self.conv2(x1)+1
        v3 = self.conv3(x1)-1
        v4 = self.conv4(x1)*0
        v5 = self.conv5(x1)*1
        v6 = self.conv6(x1)+1
        v7 = self.conv7(x1)-1
        v8 = self.conv8(x1)*0
        v9 = v4 + v5
        v10 = v5 + v6
        v11 = v6 + v7
        v12 = v7 + v8
        v13 = v3 + v12
        v14 = v4 + v9
        v15 = v5 + v10
        v16 = v6 + v11
        v17 = v7 + v12
        v18 = v8 + v9
        v19 = v3 + v18
        v20 = v14 + v16
        v21 = v16 + v18
        v22 = v18 + v21
        return v22*v19
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
