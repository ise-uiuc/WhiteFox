
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v4 = self.conv(v3)
        v5 = v4 + 3
        v6 = v4.clamp_max(6)
        v7 = v4.clamp_min(3)
        v8 = v4.mul(v5)
        v9 = v8 / 6
        v10 = v3.mul(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
