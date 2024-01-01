
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1.sub_(3)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2 / 6
        v4 = self.other_conv(v3)
        v5 = v4.add_(6)
        v6 = v5.clamp(min=0, max=6, out=None)
        v7 = v6.clamp(max=2, out=v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
