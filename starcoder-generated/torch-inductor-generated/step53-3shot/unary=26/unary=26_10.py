
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(339, 96, 4, stride=3, padding=1, dilation=4, groups=71, bias=True)
    def forward(self, x7):
        k1 = self.conv_t(x7)
        k2 = k1 > 0
        k3 = k1 * 0.373
        k4 = torch.where(k2, k1, k3)
        return k4
# Inputs to the model
x7 = torch.randn(580, 339, 10, 8)
