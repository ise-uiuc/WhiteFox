
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 1, 1, stride=3, padding=3, bias=False)
    def forward(self, x2):
        z9 = self.conv_t(x2)
        z10 = z9 > 0
        z11 = z9 * -0.484
        z12 = torch.where(z10, z9, z11)
        return z12
# Inputs to the model
x2 = torch.randn(4, 16, 49, 98)
