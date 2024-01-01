
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 375, 3, stride=2, padding=2, bias=False)
    def forward(self, x31):
        y1 = self.conv_t(x31)
        y2 = y1 > 0
        y3 = y1 * 3.786769
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x31 = torch.randn(11, 8, 23, 19)
