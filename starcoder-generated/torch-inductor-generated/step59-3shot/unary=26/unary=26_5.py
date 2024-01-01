
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 67, 1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(67, 56, 2, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x2):
        t1 = self.conv_t(x2)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        return torch.nn.functional.avg_pool2d(t5, 4)
negative_slope = 0.34
# Inputs to the model
x2 = torch.randn(64, 7, 10, 15)
