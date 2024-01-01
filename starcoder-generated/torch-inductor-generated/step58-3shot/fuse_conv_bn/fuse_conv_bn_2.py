
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x2):
        x3 = self.bn(x2)
        x4 = self.conv(x3)
        return torch.add(x1, x2)
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4)
