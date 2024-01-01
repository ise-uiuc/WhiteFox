
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, padding=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v3)
        v6 = torch.nn.functional.pad(x3, (0, 2, 0, 2))
        v7 = torch.nn.functional.pad(x4, (0, 2, 0, 2))
        v8 = self.conv1(v6)
        v9 = self.conv2(v7)
        v10 = v8 + v9
        v11 = self.bn1(v10)
        v12 = self.bn2(v10)
        v13 = v4 + v5 + v11 + v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 66, 66)
x4 = torch.randn(1, 3, 66, 66)
