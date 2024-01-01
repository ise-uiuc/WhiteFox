
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x1):
        # t1 = self.conv_t(x1)
        # t2 = t1 > 0
        # t3 = t1 * self.negative_slope
        # t4 = torch.where(t2, t1, t3)
        return torch.where(torch.gt(self.conv_t(x1), 0), self.conv_t(x1), self.negative_slope*self.conv_t(x1))
negative_slope = -0.01
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
