
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.convb = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.convb(v1)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
