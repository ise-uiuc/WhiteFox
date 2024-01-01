
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 4, stride=2, padding=(1, 2))
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=1.0, mode='nearest', recompute_scale_factor=None)
        negative_slope = 0.40926813
        v2 = x > 0
        v3 = x * negative_slope
        v4 = torch.where(v2, x, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
x2 = torch.randn(1, 4, 16, 32)
