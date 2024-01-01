
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pool = torch.nn.MaxPool2d(1)
    def forward(self, x1, x2):
        v1 = self.pool(x1)
        v2 = self.pool(x2)
        v3 = self.conv1(v1)
        v4 = self.conv2(v2)
        v5 = v3 + v4
        v6 = v5 + x1
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
