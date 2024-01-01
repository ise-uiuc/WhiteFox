
class Model(torch.nn.Module):
    def __init__(self, negative_slope, input_channels):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(input_channels, 5, 3, stride=2, padding=0, output_padding=0)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
input_channels = 576
negative_slope = -0.095
# Inputs to the model
x2 = torch.randn(1, input_channels, 32, 32)
