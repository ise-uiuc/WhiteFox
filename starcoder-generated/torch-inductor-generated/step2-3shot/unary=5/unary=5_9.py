
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 8, 4, stride=4, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.0
        v3 = v1 * 0.125
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
