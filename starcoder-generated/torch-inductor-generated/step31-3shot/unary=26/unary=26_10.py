
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 5, stride=1)
        self.conv_t = torch.nn.ConvTranspose2d(32, 16, 5, stride=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * -10.398
        x6 = torch.where(x4, x3, x5)
        return x6
# Inputs to the model
x1 = torch.randn(2, 1, 56, 56)
