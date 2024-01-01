
class Model():
    def __init__(self, negative_slope):
        self.conv_t = torch.nn.ConvTranspose1d(194, 44, 1, stride=2, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = -0.594
# Inputs to the model
x3 = torch.randn(17, 194, 10)
