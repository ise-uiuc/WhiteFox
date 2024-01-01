
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 88, 1, bias=True, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(88, 23, 3, stride=1, padding=0, bias=False)
    def forward(self, x7):
        x8 = self.conv_t(x7)
        x9 = self.conv(x8)
        x10 = x9 < 1
        x11 = x9 > 1
        x12 = x9/3
        x13 = torch.where(x10, x9, x12)
        x14 = torch.where(x11, x9, x13)
        return x14
# Inputs to the model
x7 = torch.randn(8, 128, 4, 4)
