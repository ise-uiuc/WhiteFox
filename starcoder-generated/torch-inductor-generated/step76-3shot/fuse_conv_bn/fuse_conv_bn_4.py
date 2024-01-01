
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.prelu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 9, 1, bias=False)
        self.prelu = torch.nn.PReLU()
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.bn = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 3, 9, 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
