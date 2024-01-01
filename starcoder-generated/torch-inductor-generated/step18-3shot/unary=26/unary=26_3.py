
class Model(torch.nn.Module):
    def __init__(self, negative_slope, padding, bias):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 7, 2, stride=2, padding=padding, bias=bias)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
negative_slope = -0.015
bias = False
padding = (1, 1)
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
