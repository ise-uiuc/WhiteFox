
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose3d(1, 1, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose3d(1, 1, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t1(x)
        t2 = torch.le(t1, 0.9)
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = torch.le(t5, 0.708)
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t4, t7)
        return t8
negative_slope = 0.13
# Inputs to the model
x = torch.randn(4, 1, 2, 2, 2)
