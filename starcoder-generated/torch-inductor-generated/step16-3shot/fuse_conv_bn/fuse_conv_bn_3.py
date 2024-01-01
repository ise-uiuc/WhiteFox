
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 7, 3)
        self.bn = torch.nn.BatchNorm3d(3, affine=True)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        # Fusion optimization fails on this scenario
        return s * t
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4, 4)
