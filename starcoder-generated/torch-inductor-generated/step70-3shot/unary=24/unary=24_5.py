
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 78, 12, stride=1, padding=6)
    def forward(self, x):
        negative_slope = 1.20924155
        v1 = self.conv(x)
        v2 = v1.transpose(2, 1).reshape(-1, 1, 18)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 18, 20, 20)
