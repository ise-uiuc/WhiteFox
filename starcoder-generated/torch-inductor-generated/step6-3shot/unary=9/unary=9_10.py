
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add3 = torch.add(3)
        self.clamp6min = torch.clamp_min(6)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.add3(v1)
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(self.clamp6min)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
