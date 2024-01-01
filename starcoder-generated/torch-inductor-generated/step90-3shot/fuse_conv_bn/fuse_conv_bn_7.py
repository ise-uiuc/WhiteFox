
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x31, x41):
        y = self.conv(x31)
        z = self.conv(x41)
        return self.bn(y), self.bn(z)
# Inputs to the model
x31 = torch.randn(1, 3, 3, 3)
x41 = torch.randn(1, 3, 3, 3)
