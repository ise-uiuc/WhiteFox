
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(7, 3, 3, bias=False)
        self.bn = torch.nn.BatchNorm3d(3, affine=True)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 7, 4, 4, 4)
