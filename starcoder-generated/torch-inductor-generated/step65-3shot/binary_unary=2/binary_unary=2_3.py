
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.bn(v2)
        v4 = self.pool(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 10, 10)
