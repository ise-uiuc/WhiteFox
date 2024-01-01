
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4 / 6
        v6 = v5.div(6)
        v7 = v6.div(6)
        v8 = v7.add(3)
        v9 = v8.add(1)
        v10 = v9.clamp_min(0)
        v11 = v10.clamp_max(6)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
