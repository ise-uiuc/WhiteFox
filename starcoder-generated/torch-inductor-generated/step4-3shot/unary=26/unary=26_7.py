
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 12, 2, stride=2, padding=2)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
