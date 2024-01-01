
class Model(torch.nn.Module):
    def __init__(self, negative_slope1=0.01, negative_slope2=0.01):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.negative_slope1 = negative_slope1
        self.negative_slope2 = negative_slope2
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope1
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(x2)
        v6 = v1 > 0
        v7 = v1 * self.negative_slope2
        v8 = torch.where(v2, v5, v7)
        v9 = self.conv3(x3)
        v10 = v1 > 0
        v11 = v1 * self.negative_slope2
        v12 = torch.where(v2, v9, v11)
        return v4 + v8 + v12
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 256, 256)
x3 = torch.randn(1, 3, 128, 128)
