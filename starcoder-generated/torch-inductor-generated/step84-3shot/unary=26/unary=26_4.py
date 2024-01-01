
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, output_padding=1, groups=1, dilation=1, bias=True)
    def forward(self, x57):
        v1 = self.conv_t(x57)
        v2 = v1 > 0
        v3 = v1 * -0.66669
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x57 = torch.randn(1, 1, 2, 2)
