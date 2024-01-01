
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 6, 9, padding=2, dilation=2, output_padding=1, groups=14, bias=False)
    def forward(self, x14):
        z1 = self.conv_t(x14)
        z2 = z1 > 0
        z3 = z1 * -7.604
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x14 = torch.randn(96, 64, 4, 6)
