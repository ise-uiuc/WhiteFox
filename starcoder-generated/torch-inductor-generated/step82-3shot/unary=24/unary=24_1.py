
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(63, 61, 1, stride=1)
    def forward(self, x):
        negative_slope = 0.00010363942
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = (v1 - v1)
        v4 = (v1 - v1)
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 63, 46, 70)
