
class Leaky(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Leaky, self).__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        return torch.where(x>0, x, x * self.negative_slope)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.leaky = Leaky(0.01)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = self.leaky(v2)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
