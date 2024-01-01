
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, dilation=2, padding=4, output_padding=1, groups=32, bias=False)
    def forward(self, x6):
        x1 = self.conv_t(x6)
        x2 = x1 > 0
        x3 = x1 * 0.0
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x6 = torch.randn(1, 32, 15, 34)
