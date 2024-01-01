
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = self.conv1(v2)
        v4 = self.bn2(v3)
        v5 = self.conv1(v4)
        v6 = self.bn3(v5)
        v7 = v6.add(self.bn4(v4))
        return v7
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
