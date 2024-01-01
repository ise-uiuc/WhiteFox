
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 16, 2, stride=1, padding=0, bias=False)
        self.negative_slope = negative_slope
        self.conv_t_1 = torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=0, bias=False)
    def forward(self, input0):
        y0 = input0.transpose(-1, -3)
        y1 = self.conv_t(y0)
        y2 = y1 > 0.0
        y3 = y1 * self.negative_slope
        y4 = torch.where(y2, y1, y3)
        y5 = self.conv_t_1(y4)
        return y5
# Negative slope
negative_slope1 = -0.9479
negative_slope2 = 0.0779
negative_slope3 = 0.4009
# Inputs to the model
input0 = torch.randn(16, 64, 12, 13)
