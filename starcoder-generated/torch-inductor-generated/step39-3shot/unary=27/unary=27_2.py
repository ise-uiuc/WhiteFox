
class Model(torch.nn.Module):
    def __init__(self, min, max, min_2, max_2):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.min = min
        self.max = max
        self.min_2 = min_2
        self.max_2 = max_2
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.clamp_min(v3, self.min_2)
        v5 = torch.clamp_max(v4, self.max_2)
        return v5
min = 1
max = 0
min_2 = 1
max_2 = 0
# Inputs to the model
x1 = torch.randn(1, 3, 100, 200)
