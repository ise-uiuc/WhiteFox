
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
        v6 = self.conv1(x3)
        v7 = self.conv2(x4)
        v8 = v6 + v7
        v9 = self.bn1(v8)
        v10 = self.bn2(v8)
        v11 = v4 + v5 + v9 + v10
        return v11
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
x3 = torch.randn(2, 3, 64, 64)
x4 = torch.randn(2, 3, 64, 64)
