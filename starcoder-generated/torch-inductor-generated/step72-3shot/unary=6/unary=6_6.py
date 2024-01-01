
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.averagepool = torch.nn.AvgPool2d(2, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(3, 16, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.averagepool(x1)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
