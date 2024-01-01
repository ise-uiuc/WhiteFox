
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 2)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y
# Inputs to the model
x = torch.randn(1, 1, 4, 4, 1)
