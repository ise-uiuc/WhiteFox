
class M1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, dilation):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation)
    def forward(self, x2):
        f1 = self.conv_t(x2)
        f2 = torch.relu(f1)
        return f2
class M2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1):
        t2 = self.conv_t1(x1)
        f1 = self.bn1(t2)
        g1 = self.relu1(f1)
        return g1
class M3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv_t1(x)
        x2 = self.relu1(x1)
        return x2
# Inputs to the model
x2 = torch.randn(1, 100, 8, 8)
x1 = torch.randn(1, 50, 8, 8)
