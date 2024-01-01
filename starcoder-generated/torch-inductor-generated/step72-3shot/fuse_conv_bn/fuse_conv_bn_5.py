
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(2, 2, 2)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x3):
        i2 = self.conv1(x3)
        i2 = self.bn(i2)
        i2 = self.conv1(i2)
        return i2 + i2
# Inputs to the model
x3 = torch.randn(1, 2, 2, 2, 2)
