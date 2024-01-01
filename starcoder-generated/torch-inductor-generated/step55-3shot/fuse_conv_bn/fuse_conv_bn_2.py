
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.bn = torch.nn.BatchNorm2d(1)
        self.leaky = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky(x)
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
