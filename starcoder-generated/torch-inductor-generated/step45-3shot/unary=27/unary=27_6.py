
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(64, 32, 5, stride=2, padding=4)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(self.relu(x1))
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1
max = -8
# Inputs to the model
x1 = torch.randn(1, 64, 200, 200)
