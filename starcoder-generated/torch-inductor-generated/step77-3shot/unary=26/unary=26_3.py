
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 90, 5, stride=2, padding=0, dilation=2, groups=2, bias=True)
    def forward(self, x13):
        v1 = self.conv_t(x13)
        v2 = v1 > 0
        v3 = v1 * -9.35
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x13 = torch.randn(133, 3, 53, 145)
