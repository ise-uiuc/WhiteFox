
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=3, padding=0, dilation=3, groups=2, bias=True)
    def forward(self, x9):
        v1 = self.conv_t(x9)
        v2 = v1 > 0
        v3 = v1 * -4.94
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x9 = torch.randn(2, 4, 38, 45)
