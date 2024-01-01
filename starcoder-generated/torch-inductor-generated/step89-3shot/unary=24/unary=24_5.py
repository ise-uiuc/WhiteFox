
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 2, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=0)
    def forward(self, x):
        negative_slope = -0.45553231
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.maxpool(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(4, 2, 140, 176)
