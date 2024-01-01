
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool2d = torch.nn.AvgPool2d(3, stride=1, padding=2, count_include_pad=False)
        self.conv = torch.nn.Conv2d(256, 16, 7, stride=1, padding=0)
    def forward(self, x1):
        v1 = x1
        v2 = self.avg_pool2d(v1)
        v3 = self.conv(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0, max=6)
        v6 = v3 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)
