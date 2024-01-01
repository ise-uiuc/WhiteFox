
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 0.01
        self.conv_t = torch.nn.ConvTranspose2d(33, 66, 5, stride=2)
    def forward(self, x1):
        t1 = self.conv_t(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x1 = torch.randn(4, 33, 14, 14)
