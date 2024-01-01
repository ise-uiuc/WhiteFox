
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = v1 + v2
        v6 = v3 + v4
        v7 = self.bn1(v5)
        v8 = self.bn2(v5)
        v9 = self.bn3(v6)
        v10 = self.bn4(v6)
        v11 = 0.5329149477958679 * v7.mul(v8) + 0.8366384399414062 * v9.mul(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
