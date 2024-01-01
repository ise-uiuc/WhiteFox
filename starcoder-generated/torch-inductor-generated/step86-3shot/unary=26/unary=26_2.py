
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(541, 596, 5, stride=1, padding=2, dilation=1, output_padding=0,
                                               groups=7, bias=True)
    def forward(self, x98):
        v1 = self.conv_t(x98)
        v2 = v1 > 0
        v3 = v1 * 1.07
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x98 = torch.randn(3, 541, 28, 12)
