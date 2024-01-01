
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.bn = torch.nn.BatchNorm2d(5)

    def conv_bn(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return h

    def forward(self, x):
        h = self.conv_bn(x)
        return h

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
