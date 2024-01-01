
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(3, 3, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x3):
        y3 = self.conv(x3)
        y3 = self.bn(y3)
        y3 = self.conv(y3)
        y3 = self.bn(y3)
        return y3
# Inputs to the model
x3 = torch.randn(1, 3, 3, 3, 3)
