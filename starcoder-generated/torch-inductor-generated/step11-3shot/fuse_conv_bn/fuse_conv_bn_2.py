
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return torch.add(x1, x1)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
