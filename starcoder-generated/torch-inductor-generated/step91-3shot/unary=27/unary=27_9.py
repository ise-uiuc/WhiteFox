
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(32, 4, 5, stride=1, padding=2)
        self.conv1 = torch.nn.Conv2d(32, 4, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v0 = x1
        v1 = self.relu(v0)
        v2 = self.conv(v1)
        v3 = self.conv1(v1)
        v4 = v2 + v3
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
min = 0.5
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 32, 28, 64)
