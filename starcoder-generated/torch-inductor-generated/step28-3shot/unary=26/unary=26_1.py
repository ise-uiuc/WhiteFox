
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv_t1 = torch.nn.ConvTranspose2d(10, 10, 5)
        self.conv_t2 = torch.nn.ConvTranspose2d(10, 10, 5)
        self.negative_slope = negative_slope
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t1(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t2(t8)
        t10 = t5 > 0
        t11 = t5 * self.negative_slope
        t12 = torch.where(t10, t5, t11)
        return t12
negative_slope = -0.33
# Inputs to the model
x1 = torch.randn(6, 1, 28, 28)
