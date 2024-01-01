
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.a = 3
        self.b = 6
    def forward(self, x1):
        z1 = self.conv(x1)
        z2 = z1 + self.a
        z3 = z2.clamp_min(0)
        z4 = z3.clamp_max(self.b)
        z5 = z4 / self.b
        return z5
# Inputs to the model
c1 = torch.randn(1, 3, 64, 64)
