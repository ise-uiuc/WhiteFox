
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(3)
        self.conv = torch.nn.Conv2d(1, 6, 3, stride=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.pad(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.1
max = 0.9
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
