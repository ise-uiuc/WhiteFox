
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(40, 20, 2, stride=1, padding=2)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.4058797743320465
max = 0.011841035458163738
# Inputs to the model
x1 = torch.randn(1, 40, 133)
