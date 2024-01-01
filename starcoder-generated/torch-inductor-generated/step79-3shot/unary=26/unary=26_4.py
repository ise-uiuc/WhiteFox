
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 4, stride=2, padding=1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * -0.3
        x5 = torch.where(x3, x2, x4)
        x6 = x5 * 1.45
        x7 = x6 + 0.5
        return torch.round(x7)
negative_slope = -0.3
# Inputs to the model
x1 = torch.randn(1, 4, 22, 37)
