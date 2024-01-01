
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(37, 31, 5, stride=5, groups=37, padding=2, dilation=1, bias=False)
    def forward(self, x1):
        z1 = self.conv_t(x1)
        return z1
# Inputs to the model
x1 = torch.randn(57, 37, 63, 57)
