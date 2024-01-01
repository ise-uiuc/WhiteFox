
class Model(torch.nn.Module):
    def __init__(self):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 32, 3, stride=1)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.bn(v3)
        v5 = self.relu(v4)
        v6 = self.conv1(v5)
        v7 = v6.relu()
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 128)
