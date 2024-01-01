
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=3, padding=4)
        self.min_value = min
        self.max_value = max
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_min(v2, x2)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
min = 0.2
max = 0.3
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
x2 = 0.4
