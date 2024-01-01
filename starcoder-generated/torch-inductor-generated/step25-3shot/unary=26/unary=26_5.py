
class Model(torch.nn.Module):
    def __init__(self, stride, out_channels, kernel_size, negative_slope, padding, dilation, bias):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=1, dilation=1, bias=bias)
        self.relu = torch.nn.ReLU()
        self.negative_slope = negative_slope
    def forward(self, x4):
        v1 = self.conv_transpose2d(x4)
        v2 = self.conv2d(x4)
        v3 = self.relu(v2)
        v4 = v3 > 0
        v5 = v3 * self.negative_slope
        v6 = torch.where(v4, v3, v5)
        return v6
stride = 2
out_channels = 4
kernel_size = 3
negative_slope = 0.001
padding = 2
dilation = 4
bias = True
# Inputs to the model
x4 = torch.randn(8, 3, 8, 8)
