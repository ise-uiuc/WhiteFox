
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(49, 1, 3, stride=3)
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x):
        y = self.conv(x)
        z = self.conv_t(y)
        m = z > 0
        n = z * self.negative_slope
        o = torch.where(m, z, n)
        return o
negative_slope = 5.398
# Inputs to the model
x = torch.randn(1, 49, 8, 8)
