
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 6, 7, groups=2)
    def forward(self, x):
        y0 = self.conv_t(x)
        y1 = y0 > 0
        y2 = y0 * 0.8499
        y3 = torch.where(y1, y0, y2)
        y4 = -y3 - y3
        return y4
# Inputs to the model
x = torch.randn(4, 12, 9, 14)
