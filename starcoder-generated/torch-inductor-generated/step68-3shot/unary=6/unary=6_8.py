
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 20, 2, stride=2, padding=5)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.avg_pool(v6)
        v8 = v7 + 2
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
