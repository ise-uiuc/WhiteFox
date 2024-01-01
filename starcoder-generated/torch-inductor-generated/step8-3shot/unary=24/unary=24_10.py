
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(7, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv1(x2)
        v3 = torch.cat((v1, v2), 1)
        v4 = v3 > 0
        v5 = v3 * self.negative_slope
        v6 = torch.where(v4, v3, v5)
        return v6
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 7, 64, 64)
