
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 11)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4.div(6)
        v6 = self.other_conv(v5)
        v7 = 0 + v6
        v8 = v7.clamp_min(0)
        v9 = v8.clamp_max(6)
        v10 = 1 + v9
        v11 = v10.clamp_min(0)
        v12 = v11.clamp_max(6)
        v13 = v12.div(6)
        return v13
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
