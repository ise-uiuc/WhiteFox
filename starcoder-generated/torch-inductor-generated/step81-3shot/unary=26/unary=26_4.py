
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(1280, 4096, 1)
    def forward(self, x):
        b1 = self.conv_t(x)
        b2 = b1 - 1.02
        b3 = b2 > 0
        b4 = b2 * -0.075778
        b5 = torch.where(b3, b2, b4)
        b6 = b5 - 0.7512
        b7 = b6 > 0
        b8 = b6 * -0.75001
        b9 = torch.where(b7, b6, b8)
        b10 = b9 - 0.1699
        b11 = b10 > 0
        b12 = b10 * -0.265
        b13 = torch.where(b11, b10, b12)
        return b13
# Inputs to the model
x = torch.randn(1, 1280, 7)
