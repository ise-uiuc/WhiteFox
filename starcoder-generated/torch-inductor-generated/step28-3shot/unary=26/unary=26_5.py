
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(10, 20, 3, stride=2, padding=1, output_padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(20, 30, 3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(30, 10, 3, stride=2, padding=1, output_padding=1)
        self.negative_slope = negative_slope
    def forward(self, x4):
        t1 = self.conv_t1(x4)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t3(t8)
        t10 = t9 > 0
        t11 = t9 * self.negative_slope
        t12 = torch.where(t10, t9, t11)
        return t12
negative_slope = -0.25
# Inputs to the model
x4 = torch.randn(6, 10, 8, 8)
