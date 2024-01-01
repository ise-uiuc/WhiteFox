
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 42, 6, stride=2, padding=2)
        self.conv_a = torch.nn.Conv2d(17, 9, 3, stride=1, padding=1)
        self.conv_b = torch.nn.Conv2d(9, 19, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1 + self.conv_a(v1) + self.conv_b(v1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1.0
max = 0.5
# Inputs to the model
x1 = torch.randn(1, 2, 62, 62)
