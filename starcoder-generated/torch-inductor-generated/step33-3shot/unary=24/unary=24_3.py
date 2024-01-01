
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(6, 6, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        negative_slope = 3
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 96, 96, 96)
