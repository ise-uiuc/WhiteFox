
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.add = torch.nn.quantized.FloatFunctional()
    def forward(self, x2):
        x1 = torch.ones_like(x2)
        v1 = self.conv(x1)
        v2 = self.add.add(v1, 2)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = torch.fmod(v4, 6)
        return v5
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
