
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm1d(1)
    def forward(self, x):
        c = self.conv(x)
        b = self.bn(c)
        return b
# Inputs to the model
x = torch.randn(2, 1, 3, 3)
