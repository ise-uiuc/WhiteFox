
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(120, 86, 6, stride=2, padding=1, bias=False, dilation=2)
    def forward(self, x1):
        r1 = self.conv_t(x1)
        r2 = r1 > 0
        r3 = r1 * -0.023
        r4 = torch.where(r2, r1, r3)
        return r4
# Inputs to the model
x1 = torch.randn(28, 120, 63, 1)
