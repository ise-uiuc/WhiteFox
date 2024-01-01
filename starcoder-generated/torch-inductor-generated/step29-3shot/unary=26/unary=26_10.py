
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose1d(3, 10, 7, stride=2, padding=1, output_padding=1)
        self.conv_t2 = torch.nn.ConvTranspose1d(10, 4, 5, stride=2, padding=1, output_padding=1)
        self.negative_slope = negative_slope
    def forward(self, x7):
        t1 = self.conv_t1(x7)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        t9 = t8 * 0.314150947
        return t9
negative_slope = -0.55
# Inputs to the model
x7 = torch.randn(8, 3, 109)
