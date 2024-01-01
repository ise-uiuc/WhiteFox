
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(3,2)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.bn(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 192, 128)
