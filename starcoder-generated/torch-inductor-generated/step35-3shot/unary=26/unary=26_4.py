
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = -0.92
        self.conv = torch.nn.Conv2d(4, 7, (4, 8), stride=7)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 < 0
        v3 = v1 * self.negative_slope
        v4 = v1 - v3
        v5 = torch.clamp(v4, self.negative_slope)
        v6 = torch.where(v2, v5, v4)
        return v6
# Inputs to the model
x = torch.randn(5, 4, 2, 9, device='cpu')
