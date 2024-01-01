
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(14, 25, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.01
# Inputs to the model
x3 = torch.randn(8, 25, 5)
