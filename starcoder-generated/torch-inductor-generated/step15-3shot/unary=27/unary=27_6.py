
class Model(torch.nn.Module):
    def __init__(self, min=0.88, max=0.88):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 41, 1, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.56
max = 0.56
# Inputs to the model
x1 = torch.randn(5, 16, 20, 20)
