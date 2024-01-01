
class Model(torch.nn.Module):
    def __init__(self, min_v, max_v):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(5, 15, 1, stride=2, padding=3)
        self.conv1 = torch.nn.Conv2d(1, 2, 2, stride=1, padding=1)
        self.min_v = min_v
        self.max_v = max_v
    def forward(self, x0, x1):
        v0 = self.conv0(x0)
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v0, self.min_v)
        v3 = torch.clamp_max(v2, self.max_v)
        return v3
min_v = 0.7
max_v = 0.2
# Inputs to the model
x0 = torch.randn(1, 5, 64, 64)
x1 = torch.randn(1, 1, 64, 64)
