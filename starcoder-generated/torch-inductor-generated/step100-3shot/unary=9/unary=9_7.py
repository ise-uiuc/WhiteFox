
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.const = 3
        self.clip_min = 0
        self.clip_max = 6
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(self.const)
        v3 = v2.clamp_min(self.clip_min)
        v4 = v3.clamp_max(self.clip_max)
        v5 = v4.div(self.clip_max - self.clip_min)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
