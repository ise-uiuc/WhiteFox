
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(31, 12, 6, stride=2, padding=0, bias=False)
    def forward(self, x9):
        z1 = self.conv_t(x9)
        z2 = z1 > 0
        z3 = z1 * -2.2326000699999998
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x9 = torch.randn(11, 31, 8, 11, requires_grad=False)
