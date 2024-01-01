
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(73, 61, 3, stride=1, padding=1, bias=False, dilation=1, groups=1, output_padding=0)
    def forward(self, x26):
        v1 = self.conv_t(x26)
        v2 = v1 > 0
        v3 = v1 * -0.198
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x26 = torch.randn(7, 73, 17, 25)
