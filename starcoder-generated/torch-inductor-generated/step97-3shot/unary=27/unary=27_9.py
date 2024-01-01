
class Model(torch.nn.Module):
    def __init__(self, minv, maxv):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 1, (1, 1), 1, (0, 0), 1, bias=False)
        self.minv = minv
        self.maxv = maxv
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.minv)
        v3 = torch.clamp_max(v2, self.maxv)
        return v3
minv = -0.7
maxv = 0.6
# Inputs to the model
x1 = torch.randn(1, 5, 200, 200)
