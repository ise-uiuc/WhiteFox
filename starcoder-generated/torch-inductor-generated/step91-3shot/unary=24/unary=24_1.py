
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 2, 6, stride=1, padding=7)
    def forward(self, x):
        negative_slope = 0.7378717
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 9, 9, 10)
