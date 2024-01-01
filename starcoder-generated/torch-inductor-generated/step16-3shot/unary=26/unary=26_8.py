
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(9, 16, 1, stride=1, padding=0, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(16, 32, 1, stride=2, padding=0, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(32, 1, 1, stride=1, padding=0, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x3):
        a1 = self.conv_t1(x3)
        a2 = a1 > 0
        a3 = a1 * self.negative_slope
        a4 = torch.where(a2, a1, a3)
        a5 = self.conv_t2(a4)
        a6 = a5 > 0
        a7 = a5 * self.negative_slope
        a8 = torch.where(a6, a5, a7)
        a9 = self.conv_t3(a8)
        aa = a9 > 0
        ab = a9 * self.negative_slope
        ac = torch.where(aa, a9, ab)
        return ac
negative_slope = -0.1
# Inputs to the model
x3 = torch.randn(4, 9, 8, 8)
