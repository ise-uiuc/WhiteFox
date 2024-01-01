
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 1, stride=2, padding=21, bias=True)
    def forward(self, x5):
        z1 = self.conv_t(x5)
        z2 = z1 > 0
        z3 = z1 * -0.752
        z4 = torch.where(z2, z1, z3)
        return torch.nn.functional.interpolate(z4, scale_factor=[1.0, 1.0])
# Inputs to the model
x5 = torch.randn(3, 1, 7, 31)
