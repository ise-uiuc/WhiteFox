
class Model(torch.nn.Module):
    def __init__(self, low, hi):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 2, padding=2)
        self.low = low
        self.hi = hi
    def forward(self, x1):
        v1 = self.conv(x1)
        y = torch.clamp_min(v1, self.low)
        y = torch.clamp_max(y, self.hi)
        return y
low = 0.9
hi = 0.6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 128)
