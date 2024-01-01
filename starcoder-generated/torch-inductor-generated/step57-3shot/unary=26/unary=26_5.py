
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 64, 4, stride=2, padding=0, output_padding=0, bias=False)
    def forward(self, x16):
        h1 = self.conv_t(x16)
        h2 = h1 > 0
        h3 = h1 * 0.174
        h4 = torch.where(h2, h1, h3)
        return h4
# Inputs to the model
x16 = torch.randn(5, 12, 11, 19)
