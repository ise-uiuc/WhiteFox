
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = self.bn(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
