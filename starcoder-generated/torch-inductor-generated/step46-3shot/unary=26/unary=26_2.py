
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(61, 72, 2, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(72)
    def forward(self, x86):
        z86 = self.conv_t(x86)
        z87 = self.bn1(z86)
        z88 = z87 > 0
        z89 = z87 * -1.089
        z90 = torch.where(z88, z87, z89)
        return z90
# Inputs to the model
x86 = torch.randn(1, 61, 87, 33)
