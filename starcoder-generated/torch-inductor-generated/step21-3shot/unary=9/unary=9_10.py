
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.other_conv(v1.add(3).clamp(min=0, max=6).div(6))
        v3 = self.other_conv(v2.add(3).clamp(min=0, max=6).div(6))
        v4 = 3 + v3
        v5 = v4.clamp(min=0, max=6)
        v6 = v5 / 6
        v7 = self.other_conv(v6)
        v8 = self.other_conv(v7.add(3).clamp(min=0, max=6).div(6))
        v9 = 3 + v8
        v10 = v9.clamp(min=0, max=6)
        v11 = v10 / 6
        return v11
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
