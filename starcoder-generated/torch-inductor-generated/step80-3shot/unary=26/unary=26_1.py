
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(115, 73, 3, stride=1, padding=1)
    def forward(self, x5):
        x6 = self.conv_t(x5)
        x7 = x6 > 0
        x8 = x6 * -0.4032
        x9 = torch.where(x7, x6, x8)
        return x9 / x9
# Inputs to the model
x5 = torch.randn(11, 115, 43, 63)
