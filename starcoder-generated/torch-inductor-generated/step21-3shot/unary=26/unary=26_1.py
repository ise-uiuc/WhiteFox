
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 66, 2, stride=2, padding=0)
        self.negative_slope = 10000
    def forward(self, x5):
        x6 = self.conv_t(x5)
        x7 = x6 > 0
        x8 = x6 * self.negative_slope
        x9 = torch.where(x7, x6, x8)
        return x9
# Inputs to the model
x5 = torch.randn(3, 19, 4, 4)
