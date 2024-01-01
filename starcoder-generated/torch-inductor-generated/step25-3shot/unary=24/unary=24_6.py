
class ConvModule(torch.nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size)
        self.bn = torch.nn.BatchNorm2d(channels)
    def forward(self, x):
        x = torch.nn.functional.pad(x, [0, 0, 0, 0, self.padding, self.padding, self.padding, self.padding])
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        return x
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = ConvModule(12, 3)(x)
        return x, 0
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)
