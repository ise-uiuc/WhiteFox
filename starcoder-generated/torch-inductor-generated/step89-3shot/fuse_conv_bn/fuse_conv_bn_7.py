
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(56, 62, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(56)
    def customForward(self, x):
        a = self.conv(x)
        b = self.bn(x)
        y = self.conv(b)
        y = self.bn(y)
        return y
    def forward(self, x):
        y = self.customForward(x)
        y = self.customForward(y)
        y = self.customForward(y)
        return y
# Inputs to the model
x = torch.randn(1, 56, 56, 56)
