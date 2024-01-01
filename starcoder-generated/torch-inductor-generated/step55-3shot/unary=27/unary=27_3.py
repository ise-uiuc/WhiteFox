
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv2d(64, 64, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 77.30905532836914
max = 999.2732543945312
# Inputs to the model
x1 = torch.randn(2, 64, 128, 321)
