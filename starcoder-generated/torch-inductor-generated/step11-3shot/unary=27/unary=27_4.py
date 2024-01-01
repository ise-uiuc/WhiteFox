
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.t1 = torch.nn.Conv2d(19, 19, 3, stride=3, padding=16)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.5
max = -2
# Inputs to the model
x1 = torch.randn(1, 19, 64, 64)
