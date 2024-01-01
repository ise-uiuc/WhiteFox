
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1, groups=8)
    def forward(self, x1):
        v2 = self.conv(x1) + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 2)
        v5 = v4 / 2
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
