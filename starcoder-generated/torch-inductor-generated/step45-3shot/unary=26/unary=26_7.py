
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(78, 3, 2, stride=1, padding=0, bias=False)
    def forward(self, x3):
        g1 = self.conv_t(x3)
        g2 = g1 > 0
        g3 = g1 * -0.25
        g4 = torch.where(g2, g1, g3)
        return g4
# Inputs to the model
x3 = torch.randn(31, 78, 75, 21)
