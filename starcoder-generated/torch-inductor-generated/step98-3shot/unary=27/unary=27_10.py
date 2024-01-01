
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.add = torch.add
        self.clamp = torch.clamp_min
        self.conv_p = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.clamp(v1, self.min)
        v3 = self.conv(v2)
        v4 = self.add(v3, self.min)
        v5 = self.clamp(v4, self.min)
        v6 = self.conv_p(v5)
        v7 = self.clamp(v6, self.max)
        return v7
min = 0.9
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
