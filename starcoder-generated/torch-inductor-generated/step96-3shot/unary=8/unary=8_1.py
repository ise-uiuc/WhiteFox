
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups=1):
        super(ConvBN, self).__init__()
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_planes)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBN(1, 16, 5, stride=2, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, kernel_size=4, stride=4)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_transpose(x1)
        x3 = x2 + 3
        x4 = torch.clamp(x3, min=0)
        x5 = torch.clamp(x4, max=6)
        x6 = x1 * x5
        x7 = x6 / 6
        return x7


# Inputs to the model
x1 = torch.randn(1, 1, 30, 30)
