:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 1, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(1, 5, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = -0.82340765
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = self.conv2(v4)
        v5 = v4 > 0
        v6 = v4 * negative_slope
        v7 = torch.where(v2, v1, v3)
        v8 = torch.where(v5, v4, v6)
        return v8
# Inputs to the model
x1 = torch.randn(2, 4, 128, 84)
