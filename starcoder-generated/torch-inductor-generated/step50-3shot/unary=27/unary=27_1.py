
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.padding = torch.nn.ConstantPad2d(padding=(8, 8, 8, 8), value=0.)
        self.conv = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.padding(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -2
max = 1
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
