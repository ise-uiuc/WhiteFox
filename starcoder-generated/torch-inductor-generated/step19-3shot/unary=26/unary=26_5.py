
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * self.negative_slope
        x6 = torch.where(x4, x3, x5)
        return x6
negative_slope = 10000
# Inputs to the model
x2 = torch.randn(8, 19, 4, 4)
