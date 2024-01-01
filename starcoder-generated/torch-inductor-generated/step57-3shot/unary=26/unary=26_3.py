
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 80, 3, stride=2, padding=1, bias=False)
    def forward(self, x14):
        y1 = self.conv_t(x14)
        y2 = y1 > 0
        y3 = y1 * 4.108
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x14 = torch.randn(3, 19, 7, 9)
