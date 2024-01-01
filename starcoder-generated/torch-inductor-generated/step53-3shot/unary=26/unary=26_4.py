
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(242, 317, 2, stride=1, padding=0)
    def forward(self, x0):
        y0 = self.conv_t(x0)
        y1 = y0 > 0
        y2 = y0 * -0.260
        y3 = torch.where(y1, y0, y2)
        return y3
# Inputs to the model
x0 = torch.randn(3, 242)
