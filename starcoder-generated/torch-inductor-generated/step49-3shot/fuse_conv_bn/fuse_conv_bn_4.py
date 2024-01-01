
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(16, 32, 3, stride=2)
        self.bn = torch.nn.BatchNorm3d(16)
    def forward(self, x):
        y = self.bn(x)
        y = self.conv(y)
        return y
# Inputs to the model
x = torch.randn(1, 16, 3, 3, 3)
