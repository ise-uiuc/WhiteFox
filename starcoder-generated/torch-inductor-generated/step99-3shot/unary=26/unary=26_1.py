
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(58, 35, 2, stride=2, padding=0, output_padding=1, bias=False)
    def forward(self, o1):
        c1 = self.conv_t(o1)
        c2 = c1 > 0
        c3 = c1 * 0.447
        c4 = torch.where(c2, c1, c3)
        return c4
# Inputs to the model
o1 = torch.randn(40, 58, 85, 98)
