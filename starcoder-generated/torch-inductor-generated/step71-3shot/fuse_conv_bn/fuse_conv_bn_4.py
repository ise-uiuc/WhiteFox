
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(5, 3, 1, bias=False)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.conv(y)
        return y
# Inputs to the model
x = torch.randn(1, 5, 6, 8, 9)
