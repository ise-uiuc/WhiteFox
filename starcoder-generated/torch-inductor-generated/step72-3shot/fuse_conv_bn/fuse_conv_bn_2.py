
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(32, 16, 5, stride=3, padding=1, groups=3, bias=True)
        self.bn = torch.nn.BatchNorm3d(16)
    def forward(self, x3):
        r3 = self.conv(x3)
        r3 = self.bn(r3)
        r3 = self.conv(r3)
        r3 = self.bn(r3)
        r3 = self.conv(r3)
        r3 = self.bn(r3)
        r3 = self.conv(r3)
        r3 = self.bn(r3)
        return r3 + r3
# Inputs to the model
x3 = torch.randn(3, 32, 32, 32, 32)
