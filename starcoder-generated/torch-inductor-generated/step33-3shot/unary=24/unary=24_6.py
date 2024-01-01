
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        negative_slope = 1
        v2 = v1 < 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
