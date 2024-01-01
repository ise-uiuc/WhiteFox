
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 2, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 + self.negative_slope
        t3 = t2 > 0
        t4 = t2 * self.negative_slope
        t5 = torch.where(t3, t2, t4)
        return t1 + t5
negative_slope = 0.91
# Inputs to the model
x = torch.randn(16, 2, 2, 2)
