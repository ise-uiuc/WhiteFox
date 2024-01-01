
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(33, 8, 11, stride=5, padding=5, dilation=1, groups=3)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1, dilation=1, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.add(v1, 3)
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = torch.div(v3, 6)
        v5 = self.other_conv(v4)
        v6 = torch.add(v5, 0)
        v7 = v6.clamp(min=0, max=6)
        v8 = torch.div(v7, 6)
        return v8
# Inputs to the model
x1 = torch.randn(9, 33, 64, 64)
