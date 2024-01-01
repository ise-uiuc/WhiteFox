
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 25, 6, stride=2, padding=3, dilation=3, groups=5)
    def forward(self, v1):
        v3 = self.conv_t(v1)
        v4 = v3 > 0
        v5 = v3 * -14.94
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
v1 = torch.randn(1, 40, 15, 17)
