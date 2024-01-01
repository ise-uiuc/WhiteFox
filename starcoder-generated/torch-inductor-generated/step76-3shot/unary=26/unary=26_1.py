
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(150, 106, 1, stride=1, padding=0)
    def forward(self, x1):
        h1 = self.conv_t(x1)
        h2 = h1
        h3 = h1
        h4 = h2 * h3
        h5 = h1
        h6 = h5
        h7 = h6
        h8 = h1
        h9 = h8
        h10 = h1
        h11 = h10
        return h1, h4, h7, h9, h11
# Inputs to the model
x1 = torch.randn(10, 150, 32, 32)
