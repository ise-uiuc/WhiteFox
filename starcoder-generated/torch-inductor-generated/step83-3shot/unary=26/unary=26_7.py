
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 1)
    def forward(self, input):
        y1 = self.conv(input)
        y1 += 1
        y2 = self.conv_t(y1)
        y3 = y2 > 0
        y4 = y2 * -0.35
        y5 = torch.where(y3, y2, y4)
        y5[0, :] -= 1
        return y5
# Inputs to the model
input = torch.randn(1, 3, 10, 10)
