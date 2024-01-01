
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(42, 19, 1, stride=1, padding=0, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x6):
        x7 = self.conv_t(x6)
        x8 = x7 > 0
        x9 = x7 * self.negative_slope
        x10 = torch.where(x8, x7, x9)
        return torch.nn.functional.interpolate(x10, scale_factor=[2.0, 1.0])
negative_slope = 0.0
# Inputs to the model
x6 = torch.randn(1, 42, 2, 3)
