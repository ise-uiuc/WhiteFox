
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)
    def forward(self, x7):
        z1 = self.conv_t(x7)
        z2 = z1 > 0
        z3 = z1 * -10
        z4 = torch.where(z2, z1, z3)
        return torch.nn.functional.elu(z4, 0.01968836506661952)
# Inputs to the model
x7 = torch.randn(3, 1, 36, 44)
