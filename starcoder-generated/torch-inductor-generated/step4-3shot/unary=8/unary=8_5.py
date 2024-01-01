
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 10, 3, padding=1, groups=32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3.0
        v3 = v2 / 100
        v4 = torch.clamp(v3, min=-5, max=6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
