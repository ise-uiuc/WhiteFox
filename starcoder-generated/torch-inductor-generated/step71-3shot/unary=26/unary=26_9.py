
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(22, 55, 4, stride=2, padding=1)
    def forward(self, x11):
        a1 = self.conv_t(x11)
        a2 = a1 > 0
        a3 = a1 * -0.419
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x11 = torch.randn(31, 22, 11, 9)
