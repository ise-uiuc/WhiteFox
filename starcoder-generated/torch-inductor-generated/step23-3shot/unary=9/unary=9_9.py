
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, 3, stride=3, padding=1)
        self.other_conv = torch.nn.Conv2d(9, 9, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 7 + v1
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4 / 6
        v6 = self.other_conv(v5)
        v7 = 7 + v6
        v8 = v7.clamp_min(0)
        v9 = v8.clamp_max(6)
        v10 = v9 / 6
        return v10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
