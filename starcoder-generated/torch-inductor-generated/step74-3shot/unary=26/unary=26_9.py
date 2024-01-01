
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 128, 1, stride=1, bias=False)
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 1, 1, stride=1)
    def forward(self, x11):
        y1 = self.conv_t(x11)
        y2 = self.conv(y1)
        y3 = y2 > 0
        y4 = y2 * -1.1
        y5 = torch.where(y3, y2, y4)
        y6 = self.conv2(y5)
        return y6
# Inputs to the model
x11 = torch.randn(1, 2, 112, 112)
