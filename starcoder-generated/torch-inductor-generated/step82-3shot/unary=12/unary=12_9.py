
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn1(v1)
        v3 = self.relu1(v2)
        v4 = self.bn2(v3)
        v5 = self.relu2(v4)
        v6 = self.conv1(v5)
        x = torch.sigmoid(v6)
        return x
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
