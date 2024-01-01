
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        t0 = (1.0, 1.0, 1.0, 6.0)
        v2 = v1 + t0
        v3 = torch.clamp_min(v2, 0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
