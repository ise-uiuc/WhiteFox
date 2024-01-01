
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        negative_slope = torch.randn(459)
        v1 = torch.conv1d(x, weight, stride=2, padding=2, dilation=1,
                            groups=1)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(45, 32, 15)
weight = torch.randn(32, 45, 12)
