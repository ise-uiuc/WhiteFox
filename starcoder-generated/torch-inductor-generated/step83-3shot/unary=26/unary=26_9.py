
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(63, 2, 5, stride=1, padding=2, dilation=1)
        self.conv_t = torch.nn.ConvTranspose2d(2, 36, 5, stride=1, padding=2, dilation=1)
    def forward(self, x1):
        x2 = torch.nn.functional.gelu(self.conv(x1))
        x3 = torch.nn.functional.gelu(self.conv_t(x2))
        x4 = 0.5 * x3
        x5 = x4 > 0
        x6 = x4 * 0.1349
        x7 = torch.where(x5, x4, x6)
        return x7
# Inputs to the model
x1 = torch.randn(2, 63, 44, 11)
