
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 13, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = (x2 + 3)
        x4 = torch.clamp_max(x3, 6)
        x5 = (x2 + 3)
        x6 = torch.clamp_max(x5, 6)
        x7 = x4 * x6
        x8 = x7 / 6
        x9 = torch.ones_like(x8)
        x10 = torch.ones_like(x8)
        x11 = x10 * 6
        x12 = x8 + x11
        x13 = torch.clamp_min(x12, 0)
        return x13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
