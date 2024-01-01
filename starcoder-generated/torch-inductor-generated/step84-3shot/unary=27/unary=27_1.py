
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.min
        if v2 is not None:
            v1 = torch.clamp_min(v1, v2)
        out = v1
        v2 = self.max
        if v2 is not None:
            v1 = torch.clamp_max(out, v2)
        out = v1
        return out
min = 1.0
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
