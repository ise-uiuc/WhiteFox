
class Model(torch.nn.Module):
    def __init__(self, min_, max_):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 4, stride=2, padding=1)
        self.min_ = min_
        self.max_ = max_
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_)
        v3 = torch.clamp_max(v2, self.max_)
        return v3
min_ = 0.6
max_ = 0.8
# Inputs to the model
x1 = torch.randn(1, 1, 200, 200)
