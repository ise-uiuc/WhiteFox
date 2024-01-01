
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(556, 320, 5, stride=1)
        self.conv_t = torch.nn.ConvTranspose2d(320, 88, 5, stride=1)
    def forward(self, x11):
        x12 = self.conv(x11)
        x13 = self.conv_t(x12)
        x14 = x13 > 0
        x15 = x13 * 0.1393
        x16 = torch.where(x14, x13, x15)
        return x16
# Inputs to the model
x11 = torch.randn(111, 556, 20, 3)
