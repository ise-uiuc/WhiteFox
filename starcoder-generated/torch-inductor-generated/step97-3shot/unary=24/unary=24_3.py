
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 2, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 3, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.710689
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 12, 44, 64)
