
class Model(torch.nn.Module):
    def __init__(self, shape_a, shape_b, shape_c, shape_d):
        super().__init__()
        self.conv = torch.nn.Conv2d(shape_a, shape_c, shape_d[2], stride=2, padding=shape_b)
    def forward(self, x):
        negative_slope = 2.8031635
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
