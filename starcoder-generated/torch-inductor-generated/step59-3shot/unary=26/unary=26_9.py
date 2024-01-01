
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(120, 112, 2, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose1d(112, 80, 3)
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * -0.32
        x6 = torch.where(x4, x3, x5)
        x7 = self.conv_t2(x6)
        x8 = x7 > 0
        x9 = x7 * -0.1
        x10 = torch.where(x8, x7, x9)
        return x10
# Inputs to the model
x2 = torch.randn(2, 120, 60)
