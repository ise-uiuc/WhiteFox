
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1, dilation=1, groups=2)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=2, padding=1, dilation=1, groups=2)
    def forward(self, x):
        negative_slope = 0.1
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = v4 > 0
        v6 = v4 * 0.1
        v7 = torch.where(v5, v4, v6)
        v8 = self.conv2(v7)
        v9 = v8 > 0
        v10 = v8 * 0.1
        v11 = torch.where(v9, v8, v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
