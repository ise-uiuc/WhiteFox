
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3.add(v4)
        v6 = self.conv3(v5)
        v7 = self.conv4(v5)
        v8 = self.bn3(v6)
        v9 = self.bn4(v7)
        v10 = v8 - v9
        v11 = v8.mul(v9)
        return v10.div(v11)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
