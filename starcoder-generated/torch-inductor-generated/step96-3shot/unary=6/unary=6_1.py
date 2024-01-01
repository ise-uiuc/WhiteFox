
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 1, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=11, ceil_mode=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(16)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 32)
        v5 = v1.mul(v4)
        v6 = v5 / 32
        v7 = self.pool(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
