
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 17, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        t1 = torch.clamp_min(v1, self.min)
        t2 = torch.clamp_max(t1, self.max)
        v3 = torch.clamp_max(t2, self.max)
        return v3
min = 0.2
max = 0.95
# Inputs to the model
x1 = torch.randn(1, 8, 56, 56)
