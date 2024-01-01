
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 9, 2, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 4.357102
        v1 = self.conv(x)
        v2 = v1 * negative_slope
        v3 = torch.minimum(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 88, 61)
