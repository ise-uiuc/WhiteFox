
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv_t = torch.nn.ConvTranspose2d(1, 250, 95, stride=145)
        self.negative_slope = negative_slope
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = self.conv_t(v1)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
negative_slope = 1.386
# Inputs to the model
x4 = torch.randn(3, 1, 23, 122)
