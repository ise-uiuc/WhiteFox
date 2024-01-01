
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 8, 3, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.squeeze(v3, dim=1)
        return v4
min = 1.0
max = 1.0
# Inputs to the model
x1 = torch.randn(4, 5, 32, 32)
