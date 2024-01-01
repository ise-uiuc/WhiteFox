
class Model(torch.nn.Module):
    def __init__(self, min_, max_):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 8, stride=1, padding=0)
        self.min_ = min_
        self.max_ = max_
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_)
        v3 = torch.clamp_max(v2, self.max_)
        return v3
min = -0.4
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 2, 15, 15)
