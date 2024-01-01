
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(133, 34, 3, stride=2, padding=2, output_padding=1)
    def forward(self, x24):
        y1 = self.conv_t(x24)
        y2 = y1 > 0.948
        y3 = y1 * -0.258
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x24 = torch.randn((3, 133, 42, 29))
