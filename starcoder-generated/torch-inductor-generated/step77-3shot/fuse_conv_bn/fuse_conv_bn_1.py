
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4, affine=False)
    def forward(self, x1):
        y1 = self.conv(x1)
        return self.bn(y1)
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
