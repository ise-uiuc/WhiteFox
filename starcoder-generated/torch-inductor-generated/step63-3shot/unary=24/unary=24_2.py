
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(98, 75, (3, 4), stride=1, dilation=1, groups=11, padding=12)
        self.conv1 = torch.nn.Conv2d(98, 75, (3, 4), stride=1, dilation=1, groups=11, padding=12)
    def forward(self, x):
        negative_slope = 2.778335
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv1(x)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 98, 17, 3)
