
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = torch.nn.functional.pad(v3 + v4, (0, -1, -1, -1))
        v6 = self.conv3(v3.view(1, 1, -1, -1))
        v7 = self.conv4(v4.view(1, 1, -1, -1))
        v8 = self.bn3(v5)
        v9 = self.bn4(v5)
        v10 = v6 + v7
        v11 = self.bn3(v10)
        v12 = self.bn4(v10)
        v13 = v11.mul(v12)
        v14 = v11 - v13
        return v13 + v14
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
