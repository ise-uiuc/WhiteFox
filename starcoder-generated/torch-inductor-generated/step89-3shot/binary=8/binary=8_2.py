
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride=1, padding=1, groups=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.bn1(v1)
        v4 = v2 + v3
        v5 = self.bn2(v4)
        return v5
# Inputs to the model
x = torch.randn(2, 32, 128, 64)
