
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        negative_slope = 0.1
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
