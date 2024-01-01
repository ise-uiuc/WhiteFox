
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 1e-10
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
