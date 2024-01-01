
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 74, 5, stride=2, padding=2, bias=True)
    def forward(self, x34):
        y1 = self.conv_t(x34)
        y2 = y1 > -4.44
        y3 = y1 * -2.384
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x34 = torch.randn(31, 19, 25, 78)
