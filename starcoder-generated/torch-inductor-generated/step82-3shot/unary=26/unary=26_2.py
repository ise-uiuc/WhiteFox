
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(31, 5, 2, bias=False, stride=2)
    def forward(self, x13):
        x11 = self.conv_t(x13)
        u14 = x11 > 0
        u15 = x11 * 0.4779116923473358
        u16 = torch.where(u14, x11, u15)
        x17 = torch.nn.functional.rrelu(u16, 0.21, (False, True))
        x18 = torch.nn.functional.rrelu(u16, 0.43, (False, False))
        return torch.nn.functional.rrelu(x17, 3.36, (True, False))
# Inputs to the model
x13 = torch.randn(1, 31, 14, 39)
