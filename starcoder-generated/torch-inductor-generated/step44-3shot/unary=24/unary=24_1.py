
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 1.811989
        v1 = self.avgpool(x)
        v2 = self.conv(x1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Input to the model
x1 = torch.randn(1, 3, 10, 10)
