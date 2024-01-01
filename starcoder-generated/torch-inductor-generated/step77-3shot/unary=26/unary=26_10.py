
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 32, 4, stride=1, padding=0, groups=1, bias=False)
    def forward(self, x11):
        v1 = self.conv_t(x11)
        v2 = v1 > 0
        v3 = v1 * 5.3506
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x11 = torch.randn(2, 3, 10, 7)
