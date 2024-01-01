
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.bn = torch.nn.BatchNorm1d(1)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
