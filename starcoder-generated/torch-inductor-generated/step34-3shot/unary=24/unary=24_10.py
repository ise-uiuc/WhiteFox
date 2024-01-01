
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 2.0
        self.conv = torch.nn.Conv2d(10, 2, 7, stride=1, padding=7)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
