
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 78, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(78, 13, 5, groups=78, padding=2)
    def forward(self, x):
        negative_slope = 0.70576613
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 33, 87)
