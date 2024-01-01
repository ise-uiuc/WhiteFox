
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(96, 3, 67, stride=1, padding=52, bias=False)
    def forward(self, x8):
        x1 = self.conv_t(x8)
        x2 = x1 > 0
        x3 = x1 * -0.801
        x4 = torch.where(x2, x1, x3)
        return torch.nn.AdaptiveAvgPool2d(78)(x4)
# Inputs to the model
x8 = torch.randn(1, 96, 40, 72)
