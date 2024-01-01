
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = self.bn(v2)
        v4 = torch.relu(v3)
        v5 = self.bn2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
