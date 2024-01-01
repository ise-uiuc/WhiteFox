
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, output_padding=1, dilation=1)
        self.negative_slope = negative_slope
    def forward(self, x4):
        v4 = self.conv_transpose(x4)
        v5 = v4 > 0
        v6 = v4 * self.negative_slope
        v7 = torch.where(v5, v4, v6)
        return v7
negative_slope = -0.05786
# Inputs to the model
x4 = torch.randn(1, 3, 16, 16)
