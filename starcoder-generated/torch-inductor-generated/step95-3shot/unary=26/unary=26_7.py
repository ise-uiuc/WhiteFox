
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(29, 6, 2, stride=1, padding=0, bias=False)
    def forward(self, x49):
        b1 = self.conv_t(x49)
        b2 = b1 > 0
        b3 = b1 * 0.000
        b4 = torch.where(b2, b1, b3)
        return b4
# Inputs to the model
x49 = torch.randn(1, 29, 30, 6)
