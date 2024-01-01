
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(25, 50, 4, stride=2, padding=2, output_padding=1, groups=1, dilation=1, bias=False)
    def forward(self, x9):
        v1 = self.conv_t(x9)
        v2 = v1 > 0
        v3 = v1 * 0.396
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x9 = torch.randn(5, 25, 100)
