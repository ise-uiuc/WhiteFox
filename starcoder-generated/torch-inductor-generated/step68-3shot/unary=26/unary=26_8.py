
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(49, 10, 5, stride=1, padding=2, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.interpolate(v4, scale_factor=[1.0, 1.0, 1.0, 1.0], mode='nearest')

negative_slope = 0.09
# Inputs to the model
x5 = torch.randn(1, 49, 36, 32)
