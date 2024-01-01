
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        i1 = self.conv_t(x1)
        i2 = i1 > 0
        i3 = i1 * 0.1
        i4 = torch.where(i2, i1, i3)
        return i4
# Inputs to the model
x1 = torch.randn(1, 3, 47, 67)
