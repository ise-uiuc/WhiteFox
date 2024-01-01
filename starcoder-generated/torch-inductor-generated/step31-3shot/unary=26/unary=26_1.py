
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 8, stride=4, dilation=2, padding=13)
        self.negative_slope = 1.0
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(4, 1, 56, 56)
