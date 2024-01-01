
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, padding=1, groups=3)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x):
        y = self.conv(x)
        return self.bn(y)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
