
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(21, 46, 5, stride=3, padding=0, bias=False)
    def forward(self, x1):
        i1 = self.conv_t(x1)
        i2 = i1 > 0
        i3 = i1 * 1.183
        i4 = torch.where(i2, i1, i3)
        return i4
# Inputs to the model
x1 = torch.randn(6, 21, 27, 36)
