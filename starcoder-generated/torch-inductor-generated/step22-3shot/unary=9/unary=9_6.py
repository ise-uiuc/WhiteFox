
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=1, padding=0)
        self.other_conv = torch.nn.Conv2d(6, 12, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.other_conv(v4)
        v6 = 3 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(10, 3, 64, 64)
