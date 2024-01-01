
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 1, 2)
        self.bn = torch.nn.BatchNorm2d(16, affine=False)
    def forward(self, x5):
        x5 = self.conv(x5)
        x5 = self.bn(x5)
        return x5
# Inputs to the model
x5 = torch.randn(1, 8, 4, 4)
