
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(7, 7, 3, stride=1, padding=1, dilation=1, groups=7, bias=False)
        self.bn = torch.nn.BatchNorm2d(7, affine=True)
    def forward(self, x1):
        s1 = self.conv(x1)
        s2 = self.bn(s1)
        return s2
# Inputs to the model
x1 = torch.randn(1, 7, 4, 4, 4)
