
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24, 60, 2, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(60, 30, 2, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
negative_slope = 0.2
# Inputs to the model
x1 = torch.randn(1, 24, 60, 60)
