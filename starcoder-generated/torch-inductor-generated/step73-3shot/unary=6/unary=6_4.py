
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_in_channels = 3
        self.conv_out_channels = 3
        self.conv_kernel_size = 1
        self.conv_stride = 1
        self.conv_padding = 1
        self.conv = torch.nn.Conv2d(self.conv_in_channels, self.conv_out_channels, self.conv_kernel_size, self.conv_stride, self.conv_padding)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = 3 + v1
        v3 = v2.clamp(0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
