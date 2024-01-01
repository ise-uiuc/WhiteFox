
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        y1 = self.conv(x1)
        y2 = self.conv(x2)
        y = self.bn(y1)
        out = torch.cat(4 * [y], 1)
        return torch.add(y2, out)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 3, 1, 1)
