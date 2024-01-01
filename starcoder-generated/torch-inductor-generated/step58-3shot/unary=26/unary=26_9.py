
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 12, 2, stride=10, groups=5, padding=20, dilation=3)
    def forward(self, x31):
        v1 = self.conv_t(x31)
        v2 = v1 > 0
        v3 = v1 * -0.56
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x31 = torch.randn(2, 5, 200, 188)
