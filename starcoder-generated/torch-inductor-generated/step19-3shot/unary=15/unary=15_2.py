
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.bn = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 6, 5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(6)
        self.conv3 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
