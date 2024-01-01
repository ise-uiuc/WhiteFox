
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv3d = torch.nn.Conv3d(3, 8, 1, stride=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv3d(x1)
        v3 = v1 > 0
        v4 = v2 > 0
        v5 = v1 * self.negative_slope
        v6 = v2 * self.negative_slope
        v7 = torch.where(v3, v1, v5)
        v8 = torch.where(v4, v2, v6)
        v9 = torch.cat([v7, v8], 1)
        return v9
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
