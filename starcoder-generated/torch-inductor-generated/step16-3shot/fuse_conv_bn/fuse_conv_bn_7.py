
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3)
        self.bn = nn.BatchNorm3d(1)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 1, 2, 4, 4)
