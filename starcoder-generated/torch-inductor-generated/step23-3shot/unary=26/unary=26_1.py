
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        f1 = self.conv_t(x1)
        f2 = f1 > 0
        f3 = f1 * self.negative_slope
        f4 = torch.where(f2, f1, f3)
        return f4
negative_slope = -0.75
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
