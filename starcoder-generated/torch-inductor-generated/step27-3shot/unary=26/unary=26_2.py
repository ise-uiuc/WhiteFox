
class Model(torch.nn.Module):
    def __init__(self, negative_slope, dilation):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=dilation, groups=2, bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(16, 32, kernel_size=(2, stride), stride=(stride, 1), padding=(1, padding), groups=1, bias=False, dilation=(1, 1))
        self.tanh = torch.nn.Tanh()
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(32, 1, 1)), torch.nn.Parameter(torch.randn(32, 1, 1))])
        self.negative_slope = negative_slope
    def forward(self, x):
        y = self.conv(x)
        z = self.conv_t(y)
        m = self.tanh(z)
        o = torch.tanh(m + self.bias[0] + self.bias[1]) > self.negative_slope
        p = z * self.negative_slope
        q = torch.where(m > self.negative_slope, z, p)
        return o
negative_slope = 5.885
dilation = (1, 1)
stride = 1
padding = 0
# Inputs to the model
x = torch.randn(1, 32, 11, 14)
