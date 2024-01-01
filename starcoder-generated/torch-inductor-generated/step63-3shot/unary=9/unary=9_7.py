
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = nn.Conv2d(8, 8, 1, groups=8, bias=False)
        self.bn = nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.other_conv(v1)
        v3 = self.bn(v2.mul(0.0078125))
        return v3
    def extra_repr(self):
        return 'conv(3, 8)'

# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
