
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(120, 128, 1, stride=2, padding=0, dilation=1, groups=2)
        self.conv2 = torch.nn.ConvTranspose2d(128, 131, 1, stride=2, padding=0, dilation=1, groups=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 120, 99, 91)
