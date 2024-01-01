
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 300, 3, stride=1, padding=2, bias=False)
    def forward(self, x5):
        z1 = self.conv_t(x5)
        z2 = z1 > 0
        z3 = z1 * -0.743
        z4 = torch.where(z2, z1, z3)
        return torch.nn.functional.interpolate(torch.nn.Softplus()(z4), scale_factor=[1.0, 1.0])
# Inputs to the model
x5 = torch.randn(5, 1, 7, 33)
