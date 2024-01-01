
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 5, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 8, 5, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = v4.mul(v4)
        v8 = self.conv3(x3)
        v9 = self.conv4(x4)
        v10 = v8 + v9
        v11 = v10.relu()
        v12 = self.bn3(v11)
        return v12
# Inputs to the model
x1 = torch.randn(4, 3, 32, 32)
x2 = torch.randn(4, 3, 32, 32)
x3 = torch.randn(4, 3, 16, 16)
x4 = torch.randn(4, 3, 16, 16)
