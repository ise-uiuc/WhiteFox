
class Model(torch.nn.Module):
    def __init__(self, negative_slope=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1, m):
        v1 = self.conv(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(m, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
m = torch.randn(1, 8, 64, 64) > 0
