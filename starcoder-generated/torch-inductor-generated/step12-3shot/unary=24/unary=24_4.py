
class Model(torch.nn.Module):
    def __init__(self, negative_slope=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = (v2 > 0) * self.negative_slope
        v4 = torch.where(v2 > 0, v2, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
