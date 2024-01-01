
class Model(torch.nn.Module):
    def __init__(self, negative_slope, in_channels, out_channels, kernel_size=2, stride=3, padding=0, bias=False):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.negative_slope = negative_slope
    def forward(self, x127):
        o1 = self.conv_t(x127)
        o2 = o1 > 0
        o3 = o1 * self.negative_slope
        o4 = torch.where(o2, o1, o3)
        return o4
negative_slope = 0.09
in_channels = 509
out_channels = 143
# Inputs to the model
x127 = torch.randn(2, in_channels, 34, 40)
