
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, 4, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.bn6 = torch.nn.BatchNorm2d(32)
        self.bn7 = torch.nn.BatchNorm2d(32)
        self.bn8 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        bn1 = self.bn1(v1)
        v2 = self.conv2(bn1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v2)
        v5 = self.conv5(v2)
        v6 = self.conv6(v2)
        v7 = v3 + v4 + v5 + v6
        bn2 = self.bn3(v7)
        v8 = v2 + bn2
        v9 = self.bn4(v8)
        v10 = self.conv3(v9)
        v11 = self.conv4(v9)
        v12 = self.conv5(v9)
        v13 = self.conv6(v9)
        v14 = v10 + v11 + v12 + v13
        bn3 = self.bn5(v14)
        v15 = v9 + bn3
        v16 = self.bn6(v15)
        v17 = v15 + v16
        v18 = self.conv3(v17)
        v19 = self.conv4(v17)
        v20 = self.conv5(v17)
        v21 = self.conv6(v17)
        v22 = v19 + v20 + v21 + v22
        bn4 = self.bn7(v22)
        v23 = v17 + bn4
        v24 = self.bn8(v23)
        v25 = v23 + v24
        return v25
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
