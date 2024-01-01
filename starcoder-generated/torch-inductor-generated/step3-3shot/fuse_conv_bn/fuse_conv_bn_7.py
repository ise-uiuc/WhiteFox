
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn1(x1)
        x2 = self.conv(x1)
        x2 = self.bn2(x2)
        x3 = self.conv(x2)
        y1 = torch.cat([x1, x2, x3], 1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
