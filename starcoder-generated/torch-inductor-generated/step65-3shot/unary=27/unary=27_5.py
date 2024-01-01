
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 16, 4, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.nn.functional.relu(v3)
        return v4

min = 0
max = 60
# Inputs to the model
x1 = torch.randn(1, 10, 35, 35)
