
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stride = 2
        self.conv_input_channels = 3
        self.conv_output_channels = 2
        self.conv_kernel_size = 2
        self.conv_padding = 1
        self.conv_dilation = 1
        self.conv = torch.nn.Conv2d(self.conv_input_channels, self.conv_output_channels, self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope_parameter
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope_parameter = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
