
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0)
        self.negative_slope = 10000
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * self.negative_slope
        x6 = torch.where(x4, x3, x5)
        return x6
# Inputs to the model
x2 = torch.randn(8, 19, 4, 4)
