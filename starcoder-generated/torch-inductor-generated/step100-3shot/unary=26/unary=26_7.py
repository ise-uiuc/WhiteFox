
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(6, 2, 1, stride=1, padding=0, bias=False)
    def forward(self, x5):
        z1 = self.conv_t(x5)
        z2 = z1 > 0
        z3 = z1 * 0.868
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x5 = torch.randn(3, 6, 56, 34)
