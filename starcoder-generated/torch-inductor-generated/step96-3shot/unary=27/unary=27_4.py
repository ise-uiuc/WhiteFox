
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 10, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.864181
max = 0.877751
# Inputs to the model
x = torch.randn(1, 22, 2, 4)
