
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(214, 124, 5, stride=1, padding=0)
    def forward(self, x6):
        h1 = self.conv_t(x6)
        h2 = h1 > 0
        h3 = h1 * -0.618
        h4 = torch.where(h2, h1, h3)
        h5 = h4.max(dim=3).values
        return h5
# Inputs to the model
x6 = torch.randn(5, 214, 24, 25)
