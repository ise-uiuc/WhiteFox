
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv4(x4)
        v5 = self.conv4(x5)
        v6 = self.conv4(x6)
        v7 = v1 + v2
        v8 = v7 + v3
        v9 = v8 + v4
        v10 = v9 + v5
        v11 = v10 + v6
        v12 = self.bn1(v11)
        return v12
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
x5 = torch.randn(1, 3, 64, 64)
x6 = torch.randn(1, 3, 64, 64)
