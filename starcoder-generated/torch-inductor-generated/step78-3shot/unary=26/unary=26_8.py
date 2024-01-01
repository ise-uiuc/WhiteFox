
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 1, 17, stride=1)
    def forward(self, x6):
        y1 = self.conv_t(x6)
        y2 = y1 > 0
        y3 = y1 * -1.31
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x6 = torch.randn(6, 3, 7, 42)
