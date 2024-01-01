
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transposed = torch.nn.ConvTranspose2d(2, 8, 1, stride=1, padding=2)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv_transposed(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
negative_slope = 0.0
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
