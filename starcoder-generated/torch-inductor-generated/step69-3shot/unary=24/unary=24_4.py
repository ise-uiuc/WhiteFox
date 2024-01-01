
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(4, 10, 1, stride=1, padding=0)
        self.conv2d_2 = torch.nn.Conv2d(10, 4, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.66702157
        v1 = self.conv2d_1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2d_2(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 49, 20)
