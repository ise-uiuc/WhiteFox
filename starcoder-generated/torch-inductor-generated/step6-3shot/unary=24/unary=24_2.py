
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 64, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v2 = self.conv1(x1)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3, v2, v4)
        v6 = self.conv2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
