
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1) # 8 features, 1 convolution kernel, stride 1, padding 1
    def forward(self, x):
        negative_slope = -10
        v1 = self.conv(x) # Convolve the 32-channel input with 8 1x1 kernels
        v2 = v1 > 0
        v3 = v1 * negative_slope # Multiply each element by negative_slope
        v4 = torch.where(v2, v1, v3) # Select elements from v1 when the corresponding element is True in v2 and elements from v3 when the corresponding element is False in v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64) # 32 = (batch size) * (number of channels) * (height) * (width)
