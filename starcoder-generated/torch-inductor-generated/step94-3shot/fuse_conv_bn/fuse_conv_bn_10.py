
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.bn3 = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv(x)
        x1 = self.bn1(x)
        x1 = self.bn2(x1)
        x2 = self.bn2(x)
        x3 = self.bn1(x)
        x3 = self.bn3(x)
        return x1 + x2 + x3
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
