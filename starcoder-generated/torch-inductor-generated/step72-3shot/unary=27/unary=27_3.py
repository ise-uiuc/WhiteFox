
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x):
        v = self.conv(x)
        v = torch.clamp_max(v, self.max)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.nn.functional.relu(v)
        return v4
min = -1
max = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
