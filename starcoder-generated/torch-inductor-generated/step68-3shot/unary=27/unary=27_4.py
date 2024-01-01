
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 3, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min, out=v1)
        v3 = torch.clamp_max(v2, self.max, out=v2)
        return v3
min = 0.2
max = -0.3
# Inputs to the model
x1 = torch.randn(1, 4, 224, 224)
