
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm1d(3, affine=False)
    def forward(self, x2):
        y1 = self.bn(self.conv(x2))
        y2 = self.conv(x2)
        return y1 + y2
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4)
