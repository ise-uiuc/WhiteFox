
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 5, stride=5, padding=0, dilation=0, groups=3, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 2 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
