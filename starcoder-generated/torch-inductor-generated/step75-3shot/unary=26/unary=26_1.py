
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 64, 5, stride=(1, 1), bias=False)
    def forward(self, x16):
        x19 = self.conv_t(x16)
        x20 = x19 > 0
        x21 = x19 * 0.344
        x22 = torch.where(x20, x19, x21)
        x23 = self.conv_t(x22)
        x24 = x23 > 0
        x25 = x23 * 0.212
        x26 = torch.where(x24, x23, x25)
        x27 = self.conv_t(x26)
        x28 = x27 > 0
        x29 = x27  * 0.332
        x30 = torch.where(x28, x27, x29)
        return x30
# Inputs to the model
x16 = torch.randn(56, 16, 341, 301)
