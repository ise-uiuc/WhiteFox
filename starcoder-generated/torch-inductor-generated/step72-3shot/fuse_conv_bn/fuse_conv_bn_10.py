
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(2, 2, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x26):
        z24 = self.conv(x26)
        z25 = (z24)
        z25 = self.bn(z24)
        z25 = self.bn(z25)
        z25 = self.conv(z25)
        z26 = (z25)
        z26 = self.bn(z25)
        z27 = self.bn(z26)
        return z27
# Inputs to the model
x26 = torch.randn(1, 2, 4, 4, 15)
