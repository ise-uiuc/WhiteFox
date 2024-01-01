
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 49, 6, stride=2, padding=2)
        self.conv_copy = torch.nn.Conv2d(7, 49, 6, stride=1, padding=2)
    def forward(self, x):
        negative_slope = 1.0
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv_copy(x)
        return v4
# Inputs to the model
x1 = torch.randn(1, 7, 73, 153)
