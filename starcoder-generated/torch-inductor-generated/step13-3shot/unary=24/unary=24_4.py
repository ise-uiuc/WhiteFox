
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        negative_slope = -10
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1, group=8)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
