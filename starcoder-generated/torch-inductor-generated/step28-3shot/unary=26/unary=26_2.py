
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(20, 353, 5, stride=1, padding=1, groups=9, dilation=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(353, 20, 3, stride=2, padding=1, groups=5, dilation=3)
        self.negative_slope = negative_slope
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        return t8
negative_slope = -0.3
# Inputs to the model
x1 = torch.randn(32, 20, 16, 16)
