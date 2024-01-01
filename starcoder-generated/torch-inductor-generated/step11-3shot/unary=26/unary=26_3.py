
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 20, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.67
# Inputs to the model
x4 = torch.randn(6, 10, 8, 8)
