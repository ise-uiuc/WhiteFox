
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
        self.pad = torch.nn.ReflectionPad2d((1, 0, 1, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.pad(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
