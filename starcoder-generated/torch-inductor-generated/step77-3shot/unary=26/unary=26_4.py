
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 13, 3, stride=2, padding=0, groups=13, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(13, 16, 3, stride=1, padding=1, groups=16, bias=False)
    def forward(self, x1):
        y1 = self.conv_t1(x1)
        y2 = y1 > 0
        y3 = y1 * 0.45
        y4 = torch.where(y2, y1, y3)
        y5 = self.conv_t2(y4)
        y6 = y5 > 0
        y7 = y5 * 0.11
        y8 = torch.where(y6, y5, y7)
        return y8
# Inputs to the model
x1 = torch.randn(1, 3, 10, 12)
