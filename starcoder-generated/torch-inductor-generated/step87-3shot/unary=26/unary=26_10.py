
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(516, 470, 3, stride=1, padding=2, output_padding=1, groups=11, bias=True)
        self.negative_slope = negative_slope
    def forward(self, x2):
        i1 = self.conv_t(x2)
        i2 = i1 > 0
        i3 = i1 * self.negative_slope
        i4 = torch.where(i2, i1, i3)
        return i4
negative_slope = -0.01
# Inputs to the model
x2 = torch.randn(4, 516, 6, 9)
