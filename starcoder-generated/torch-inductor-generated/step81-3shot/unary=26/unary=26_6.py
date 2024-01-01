
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(61, 14, 1)
    def forward(self, z):
        a1 = self.conv_t(z)
        a2 = a1 > 0
        a3 = a1 * 0.93236
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
z = torch.randn(21, 61, 33, 15)
