
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(47, 227, 15, stride=1, padding=0)
    def forward(self, x):
        z1 = self.conv_t(x)
        z2 = z1 > 0
        z3 = z1 * -2.1144
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x = torch.randn(54, 47, 22, 4)
