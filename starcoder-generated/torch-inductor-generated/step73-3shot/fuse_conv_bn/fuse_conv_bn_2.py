
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x2):
        v3 = self.bn(self.conv(x2))
        v4 = self.conv(x2)
        v4 = self.conv(x2)
        return self.conv(v3)
# Inputs to the model
x2 = torch.randn(1, 2, 3, 3)
