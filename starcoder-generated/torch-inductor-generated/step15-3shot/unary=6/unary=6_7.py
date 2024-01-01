
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 + v1
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v1 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
