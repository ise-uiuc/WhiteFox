
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.conv_t1 = torch.nn.ConvTranspose2d(768, 624, 3, stride=1, dilation=3, padding=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(624, 768, 1, stride=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(768, 768, 3, stride=2)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = self.conv_t2(x2)
        x4 = self.conv_t3(x3)
        x5 = x4 > 0
        x6 = x4 * self.negative_slope
        x7 = torch.where(x5, x4, x6)
        return x7
negative_slope = -0.01
# Inputs to the model
x1 = torch.randn(16, 768, 56, 56)
