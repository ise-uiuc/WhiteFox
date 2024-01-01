
class Model(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
        self.min_factor = min_factor
        self.max_factor = max_factor
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_factor)
        v3 = torch.clamp_max(v2, self.max_factor)
        return v3
min_factor = 0.13
max_factor = -0.35
# Inputs to the model
x1 = torch.randn(1, 2, 10, 10)
