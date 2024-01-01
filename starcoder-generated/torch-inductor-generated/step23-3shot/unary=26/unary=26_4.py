
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x1):
        a1 = self.conv_t(x1)
        a2 = a1 > 0
        a3 = a1 * self.negative_slope
        a4 = torch.where(a2, a1, a3)
        return a4
negative_slope = -0.01
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
