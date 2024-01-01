
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 5, 2, 1)
        self.conv2d = torch.nn.Conv2d(100, 2, 1, 2, 1)
    def forward(self, x):
        negative_slope = 0.17976282
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2d(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 10, 200, 200)
