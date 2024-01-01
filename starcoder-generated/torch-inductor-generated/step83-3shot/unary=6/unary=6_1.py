
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.add = self.conv.add
        self.clamp_min = torch.clamp_min
        self.clamp_max = torch.clamp_max
    def forward(self, x1):
        v1 = self.add(3, self.conv(x1))
        v2 = self.clamp_min(v1, 0)
        v3 = self.clamp_max(v1, 6)
        v4 = self.conv(x1) * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
