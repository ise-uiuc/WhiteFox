
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 7, 2, padding=3, output_padding=1, bias=True)
    def forward(self, x23):
        y0 = torch.randn(1, 3, 5, 4)
        y1 = self.conv_t(x23, y0)
        y2 = y1 < 0
        y3 = y1 * -0.463
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x23 = torch.randn(4, 1, 3, 5)
