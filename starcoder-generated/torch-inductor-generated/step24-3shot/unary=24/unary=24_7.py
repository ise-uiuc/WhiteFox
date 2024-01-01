
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 30, 3, stride=2, padding=2)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 16, 60, 60)
