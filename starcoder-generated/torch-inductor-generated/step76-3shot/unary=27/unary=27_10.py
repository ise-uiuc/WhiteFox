
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 27, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.transpose(v3, 0, 1)
        v5 = torch.transpose(v4, -2, -1)
        return v5
min = 15.0
max = 10.0
# Inputs to the model
x1 = torch.randn(1, 9, 8, 5)
