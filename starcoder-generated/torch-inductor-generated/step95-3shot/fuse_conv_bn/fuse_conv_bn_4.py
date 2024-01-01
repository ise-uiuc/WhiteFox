
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, 2)
        self.bn = torch.nn.BatchNorm2d(8, affine=False)
    def forward(self, x6):
        x6 = self.bn(x6)
        x6 = self.conv(x6)
        return x6
# Inputs to the model
x6 = torch.randn(1, 8, 5, 5)
