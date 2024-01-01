
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(196, 156, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        z1 = self.conv_t(x1)
        z2 = z1 > 0
        z3 = z1 * 0.952
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x1 = torch.randn(14, 196, 14, 10)
