
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1, output_padding=2, dilation=2, groups=2)
    def forward(self, x3):
        t1 = self.conv_t(x3)
        x4 = t1 > 0
        x5 = t1 * 0.294378
        x6 = torch.where(x4, t1, x5)
        return x6
# Inputs to the model
x3 = torch.randn(16, 8, 20, 20)
