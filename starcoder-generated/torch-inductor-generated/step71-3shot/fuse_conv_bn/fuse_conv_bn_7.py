
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x):
        y = self.bn(self.conv(x))
        return y
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
