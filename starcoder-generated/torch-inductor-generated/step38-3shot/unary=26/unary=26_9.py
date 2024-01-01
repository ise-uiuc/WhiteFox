
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(192, 31, 4, padding=3, dilation=1, output_padding=1)
    def forward(self, x6):
        w1 = self.conv_t(x6)
        w2 = w1 > 0
        w3 = w1 * -0.61
        w4 = torch.where(w2, w1, w3)
        return w4
# Inputs to the model
x6 = torch.randn(2, 192, 10, 13)
