
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 1, 1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x3):
        z6 = self.conv_transpose(x3)
        z1 = z6 > 0
        z2 = z6 * self.negative_slope
        z3 = torch.where(z1, z6, z2)
        return z3
negative_slope = -0.1
# Inputs to the model
x3 = torch.randn(1, 1, 64, 64, 64)
