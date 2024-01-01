
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 3, 5, stride=1, padding=1, bias=False)
    def forward(self, x):
        i1 = self.conv_t(x)
        i2 = i1 > 0
        i3 = i1 * -0.871
        i4 = torch.where(i2, i1, i3)
        return i4
# Inputs to the model
x = torch.randn(1, 5, 11, 14)
