
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 0.1
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
