
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(16)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v3)
        v6 = self.bn3(v3)
        v7 = v4 + v5 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
x3 = torch.randn(1, 3, 16, 16)
