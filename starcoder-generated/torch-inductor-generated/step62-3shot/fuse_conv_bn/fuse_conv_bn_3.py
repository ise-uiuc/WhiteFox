
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x):
        y2 = self.bn(self.conv(x))
        return y2
# Inputs to the model
x = torch.randn(1, 3, 4, 4, 4)
