
class Model():
    def __init__(self, negative_slope):
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = -150
# Inputs to the model
x2 = torch.randn(8, 19, 4, 4)
