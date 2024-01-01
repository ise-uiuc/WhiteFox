
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(32, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        i0 = self.conv_transpose(x1)
        i1 = i0 + 3
        i2 = torch.clamp_min(i1, 0)
        i3 = torch.clamp_max(i2, 6)
        i4 = i3 / 6
        return i4
# Inputs to the model
x1 = torch.randn(1, 32, 32)
