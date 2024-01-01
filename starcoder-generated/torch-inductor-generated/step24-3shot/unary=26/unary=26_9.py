
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 6, 3, stride=3)
        self.negative_slope = 1.31
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x3 = torch.randn(8, 1, 112, 112)
