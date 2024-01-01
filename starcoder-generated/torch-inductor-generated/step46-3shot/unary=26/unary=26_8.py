
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 251, 3, stride=1, padding=1, bias=False)
    def forward(self, x0):
        z1 = self.conv_t(x0)
        z2 = z1 > 0
        z3 = z1 * -0.02285
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x0 = torch.randn(1, 1, 20, 28)
