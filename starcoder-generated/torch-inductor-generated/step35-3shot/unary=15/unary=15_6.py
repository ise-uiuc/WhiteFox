
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3), stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 16, (3, 3), stride=1, padding=1, groups=64)
        self.conv4 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=1, groups=16)
        self.bn = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.bn4 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = torch.relu(self.conv(x1))
        v2 = self.bn(v1)
        v3 = torch.relu(self.conv2(v2))
        v4 = self.bn2(v3)
        v5 = torch.relu(self.conv3(v4))
        v6 = self.bn3(v5)
        v7 = torch.relu(self.conv4(v6))
        v8 = self.bn4(v7)
        y = x1 + v8
        return y
# Inputs to the model
x1 = torch.randn(4, 3, 576, 576)
