
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(244, 3, 13, stride=1, padding=0)
    def forward(self, x1):
        z3 = self.conv_t(x1)
        z1 = z3 > 0
        z2 = z3 * -0.5253
        z4 = torch.where(z1, z3, z2)
        return z4.max(dim=3).values
# Inputs to the model
x1 = torch.randn(8, 244, 43, 55)
