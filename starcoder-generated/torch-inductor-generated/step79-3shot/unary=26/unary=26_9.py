
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=0)
        self.negative_slope = negative_slope
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = x4 > 0
        x6 = x4 * self.negative_slope
        x7 = torch.where(x5, x4, x6)
        x8 = self.sigmoid(x7)
        return x8
negative_slope = 10000
# Inputs to the model
x3 = torch.randn(4, 1, 16, 21)
