
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 132, 1, stride=2, padding=5)
    def forward(self, x):
        negative_slope = 1 - 0.123
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * (1 + negative_slope)
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
