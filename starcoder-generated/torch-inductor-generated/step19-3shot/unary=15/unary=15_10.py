
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.bn3(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
