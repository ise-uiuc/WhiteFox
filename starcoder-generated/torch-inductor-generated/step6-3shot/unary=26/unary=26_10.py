
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 20, 6, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 > negative_slope
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 3
# Inputs to the model
x1 = torch.randn(1, 10, 8, 8)
