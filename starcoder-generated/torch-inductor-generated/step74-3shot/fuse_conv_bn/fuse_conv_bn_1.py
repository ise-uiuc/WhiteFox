
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 6, 1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 14, 14, 14)
