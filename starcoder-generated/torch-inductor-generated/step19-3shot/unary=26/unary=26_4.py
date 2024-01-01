
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0, bias=False)
        self.negative_slope = 10000
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(8, 19, 4, 4)
