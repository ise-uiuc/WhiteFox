
class Mod(torch.nn.Module):
    def __init__(self,negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2,1, kernel_size=(50,1),stride=(1,1),bias=False)
        self.negative_slope = negative_slope
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = x4 > 0
        x6 = x4 * self.negative_slope
        x7 = torch.where(x5, x4, x6)
        return x7
negative_slope = 2.934e-05
# Inputs to the model
x3 = torch.randn(1, 2, 1, 1)
