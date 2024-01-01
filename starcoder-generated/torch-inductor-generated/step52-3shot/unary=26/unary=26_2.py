
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(318, 203, 3, stride=1, padding=1, bias=False)
    def forward(self, x9):
        h1 = self.conv_t(x9)
        h2 = h1 > 0
        h3 = h1 * 0.138
        h4 = torch.where(h2, h1, h3)
        return h4
# Inputs to the model
x9 = torch.randn(85195, 318, 9, 69)
