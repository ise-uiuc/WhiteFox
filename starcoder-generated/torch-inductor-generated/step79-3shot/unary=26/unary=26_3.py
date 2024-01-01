
class Model():
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 4, stride=2, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * -0.3
        x6 = torch.where(x4, x3, x5)
        x7 = x6 * 1.45
        x8 = x7 + 0.5
        return torch.round(x8)
negative_slope = 1.6
# Inputs to the model
x2 = torch.randn(1, 4, 22, 37)
