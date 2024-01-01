
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(7, 15, 3, stride=3)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        x6 = self.conv_t2(x5)
        return x6
negative_slope = 0.01
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
