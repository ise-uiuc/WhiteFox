
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 4, stride=2, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        x5 = -0.3 * x4
        x6 = x5 - 0.28
        x7 = x6 + 0.5
        return x7
negative_slope = -0.3
# Inputs to the model
x = torch.randn(1, 4, 22, 37)
