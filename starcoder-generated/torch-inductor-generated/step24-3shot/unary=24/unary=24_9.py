
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 32, stride=2, padding=3)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.negative_slope
        v2 = self.conv(x)
        v3 = v2 > 0
        v4 = v2 * v1
        v5 = torch.where(v3, v2, v4)
        return v5
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
