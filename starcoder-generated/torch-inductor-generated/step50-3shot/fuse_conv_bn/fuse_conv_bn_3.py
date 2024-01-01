
class Model(torch.nn.Module):
    def __init__(self, conv_op):
        super().__init__()
        self.conv = conv_op()
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(self.conv(y))
        return y
# Inputs to the model
x = torch.randn(1, 8, 16, 16)
