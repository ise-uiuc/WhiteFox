
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 35, 13, bias=False)
    def forward(self, x7):
        z1 = self.conv_t(x7)
        z2 = z1 > 0
        z3 = z1 * 399
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x7 = torch.randn(293, 35, 18, 256)
