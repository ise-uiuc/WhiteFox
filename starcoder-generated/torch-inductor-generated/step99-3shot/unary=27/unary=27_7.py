
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(7, 12, 4, 3, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1.3
max = 0.3
# Inputs to the model
x1 = torch.randn(1, 7, 2, 3)
