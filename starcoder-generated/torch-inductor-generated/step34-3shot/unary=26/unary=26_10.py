
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(100, 572, 3, stride=1, padding=0, groups=4, bias=True)
    def forward(self, x3):
        z1 = self.conv_t(x3)
        z2 = z1 > 0
        z3 = z1 * 3.595
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x3 = torch.randn(32, 100, 50, 72)
