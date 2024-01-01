
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.bn3 = torch.nn.BatchNorm2d(3)
        self.bn4 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.bn0(v1)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.conv1(x1)
        v8 = self.bn1(v7)
        v9 = self.conv2(x1)
        v10 = self.bn2(v9)
        v11 = v6 + v10
        v12 = torch.clamp_min(v11, 0)
        v13 = torch.clamp_max(v12, 6)
        v14 = v6 * v13
        v15 = v14 / 6
        v16 = self.conv3(x1)
        v17 = self.bn3(v16)
        v18 = self.conv4(x1)
        v19 = self.bn4(v18)
        v20 = v15 + v19
        v21 = torch.clamp_min(v20, 0)
        v22 = torch.clamp_max(v21, 6)
        v23 = v15 * v22
        v24 = v23 / 6
        return v24
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
