
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_weight = torch.nn.BatchNorm2d(32, affine=True)
        self.conv = torch.nn.Conv2d(32, 512, 1, stride=1, padding=1, groups=32)
    def forward(self, x1):
        v1 = self.bn_weight(x1)
        v2 = self.conv(v1)
        v3 = v2.sigmoid()
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
