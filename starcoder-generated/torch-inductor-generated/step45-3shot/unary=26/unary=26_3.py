
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 205, 1, padding='same', bias=True)
    def forward(self, x6):
        r1 = self.conv_t(x6)
        r2 = r1 > 0
        r3 = r1 * -97.0
        r4 = torch.where(r2, r1, r3)
        return torch.nn.functional.pad(torch.nn.functional.hardtanh(torch.nn.functional.glu(torch.nn.functional.hardtanh(r4, -96, 96), 205), -39, 39), (1, 2))
# Inputs to the model
x6 = torch.randn(86, 64, 27, 37)
# model ends