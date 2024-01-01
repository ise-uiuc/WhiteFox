

class Model(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, groups=groups)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv4(x4)
        v5 = v1 + v2
        v6 = self.bn1(v5)
        v7 = v3 + v4
        v8 = self.bn2(v7)
        return v6 + v8

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)

