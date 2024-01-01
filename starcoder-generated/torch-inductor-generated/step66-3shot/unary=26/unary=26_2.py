
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(24, 220, 5, stride=2, padding=1, dilation=2, bias=False)
    def forward(self, x4):
        x1 = self.conv_t(x4)
        x2 = x1 > 0
        x3 = x1 * -0.8
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x4 = torch.randn(2, 24, 42, 40)
