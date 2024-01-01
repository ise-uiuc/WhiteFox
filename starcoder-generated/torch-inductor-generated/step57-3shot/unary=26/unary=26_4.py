
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(120, 217, 4, stride=2, padding=3)
    def forward(self, x26):
        z1 = self.conv_t(x26)
        z2 = z1 > 0
        z3 = z1 * 1.61
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x26 = torch.randn(7, 120, 44, 75)
