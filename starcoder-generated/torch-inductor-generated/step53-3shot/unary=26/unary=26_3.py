
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(97, 70, 7, stride=6, padding=2, bias=False)
    def forward(self, x3):
        h1 = self.conv_t(x3)
        h2 = h1 > 0
        h3 = h1 * -0.509
        h4 = torch.where(h2, h1, h3)
        return h4
# Inputs to the model
x3 = torch.randn(16, 97, 53, 11)
