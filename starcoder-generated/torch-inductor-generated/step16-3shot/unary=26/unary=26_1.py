
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 78, 1, stride=2, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = -0.01
# Inputs to the model
x2 = torch.randn(16, 64, 8, 8)
