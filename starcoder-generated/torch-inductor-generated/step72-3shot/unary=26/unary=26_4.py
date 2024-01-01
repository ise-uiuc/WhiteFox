
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2469, 1611, 4, stride=1, padding=1)
    def forward(self, x3):
        z1 = self.conv_t(x3)
        z2 = z1 > 0
        z3 = z1 * -0.014182783263337755
        z4 = torch.where(z2, z1, z3)
        return z4.max(dim=3).values
# Inputs to the model
x3 = torch.randn(146, 2469, 3, 93)
