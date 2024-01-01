
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(486, 378, 4, stride=1, padding=0, bias=False)
    def forward(self, x2):
        z2 = self.conv_t(x2)
        z3 = z2 > 0
        z4 = z2 * 0
        z5 = torch.where(z3, z2, z4)
        z6 = torch.neg(z5)
        z7 = torch.flatten(z6, 1)
        return torch.neg(z7)
# Inputs to the model
x2 = torch.randn(2, 486, 125, 108, 99)
