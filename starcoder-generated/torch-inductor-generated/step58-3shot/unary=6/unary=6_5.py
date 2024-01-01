
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1, groups=1)
        self.clamp_max = torch.nn.Clamp(min=-0.0, max=6.0, inplace=True)
        self.clamp_min = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = self.clamp_min(v2)
        v4 = self.clamp_max(v3)
        v5 = v1 * v4
        v6 = v5 * 4
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
