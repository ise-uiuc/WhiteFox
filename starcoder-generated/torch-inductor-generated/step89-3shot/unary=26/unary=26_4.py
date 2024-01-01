
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 10, 16, stride=2)
        self.conv_n = torch.nn.Conv2d(10, 10, 1, stride=1)
    def forward(self, x1):
        i2 = self.conv_t(x1)
        i3 = self.conv_n(i2)
        return i3
# Input to the model
x1 = torch.randn(1, 10, 7, 7)
