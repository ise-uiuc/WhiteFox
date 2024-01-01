
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 14, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = self.conv(x)
        v2 = negative_slope > 0
        v3 = negative_slope * negative_slope
        v4 = torch.where(v2, negative_slope, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 18, 14)
