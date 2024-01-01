
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 2, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return (t4 - 0.5) * 6
negative_slope = 0.13
# Inputs to the model
x = torch.randn(8, 2, 3, 3)
