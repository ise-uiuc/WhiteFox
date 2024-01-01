
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        y = self.conv(x1)
        x2 = self.bn(y)
        return torch.add(x2, x2)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
