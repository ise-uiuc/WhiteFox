
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t623 = torch.nn.modules.linear.Linear(1, 20, False)
        self.conv = torch.nn.Conv2d(20, 20, 4, stride=1)
    def forward(self, x):
        v1 = self.t623(x)
        negative_slope = 1
        v2 = self.conv(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 20, 77, 54)
