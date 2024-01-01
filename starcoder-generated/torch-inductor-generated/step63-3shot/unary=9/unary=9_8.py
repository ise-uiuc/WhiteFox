
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv2(v4)
        v6 = v5 + 3
        v7 = v6.clamp(min=0)
        v8 = v7.clamp_max(0.35294089137582703)
        v9 = v8.clamp_min(0.0039215688593685627)
        return v9
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
