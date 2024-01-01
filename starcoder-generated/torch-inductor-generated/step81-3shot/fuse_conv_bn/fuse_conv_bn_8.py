
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
        out_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias = 1, 1, 1, 1, 1, 1, 1, None
        self.bn = torch.nn.BatchNorm2d(out_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x5):
        z2 = self.conv(x5)
        return self.conv1(self.bn(z2))
# Inputs to the model
x5 = torch.randn(1, 1, 1, 1)
