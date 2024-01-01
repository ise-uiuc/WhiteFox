
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(321, 721, 2, stride=4, padding=3, bias=False)
    def forward(self, w_0):
        z1 = self.conv_t(w_0)
        z2 = z1 > -1.626
        z3 = z1 * -0.707
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
w_0 = torch.randn(3, 321, 10, 12)
