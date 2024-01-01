
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 8, stride=1, padding=2)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 0.64
max = 0.92
# Inputs to the model
x1 = torch.randn(1, 2, 53, 320)
