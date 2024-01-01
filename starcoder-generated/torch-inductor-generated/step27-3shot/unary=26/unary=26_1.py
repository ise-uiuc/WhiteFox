
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 8, stride=8, padding=8)
        self.conv_t = torch.nn.ConvTranspose2d(8, 14, 7, stride=7, padding=7)
        self.negative_slope = negative_slope
    def forward(self, x, weight):
        y = self.conv(x)
        z = self.conv_t(y)
        m = z > 0
        n = z * self.negative_slope
        o = torch.where(m, z, n)
        return o, weight
negative_slope = 0.007
# Inputs to the model
x = torch.randn(1, 6, 1, 1)
weight = torch.randn(7, 7)
