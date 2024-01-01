
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(3, 16, (7, 3), 1, (1, 0))
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv12 = torch.nn.Conv2d(16, 32, (3, 5), 1, (2, 3))
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.maxpool1 = torch.nn.MaxPool2d((3, 3), (1, 2))
        self.conv21 = torch.nn.Conv2d(32, 64, 3, 1, 0)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv22 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.maxpool2 = torch.nn.MaxPool2d((3, 4), (1, 1))
        self.conv31 = torch.nn.Conv2d(128, 256, 3, 1, 1, groups=128)
        self.conv32 = torch.nn.Conv2d(256, 256, 3, 1, 0, groups=128)
    def forward(self, x1, other=None, padding1=None, padding2=None, padding3=None):
        v1 = self.conv11(x1)
        v2 = self.conv12(v1)
        v3 = v2 + padding1
        v4 = self.bn1(v3)
        v5 = self.conv21(v4)
        v6 = self.conv22(v5)
        v7 = v6 + padding2
        v8 = self.bn2(v7)
        v9 = self.maxpool1(v8)
        v10 = self.conv31(v9)
        v11 = self.conv32(v10)
        v12 = torch.nn.functional.relu(v11) + padding3
        v13 = self.bn3(v12)
        v14 = self.bn4(v13)
        v15 = self.maxpool2(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
