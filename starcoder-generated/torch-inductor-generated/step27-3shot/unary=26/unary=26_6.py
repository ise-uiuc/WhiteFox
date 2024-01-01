
class Model(torch.nn.Module):
    def __init__(self, negative_slope=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(123, 1, 1, stride=0)
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 2, stride=0, output_padding=0)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.negative_slope if self.negative_slope else 0.01
        v3 = self.conv_t(v1)
        v4 = v3 >= 0
        v5 = v3 * v2
        v6 = torch.where(v4, v3, v5)

        return v6
negative_slope = 2.264117904944421
# Inputs to the model
x = torch.randn(1, 123, 18, 18)
