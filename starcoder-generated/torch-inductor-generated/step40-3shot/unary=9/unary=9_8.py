
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp_(min=0, max=6)
        v4 = v3.div(6)
        v5 = self.other_conv(v4.add(3).clamp_min(0).clamp_max(6).div(6))
        v6 = v5.add(3)
        v7 = v6.clamp_min(0)
        v8 = v7.clamp_max(6)
        v9 = v8.div(6)
        return v9
# Inputs to the model
x1 = torch.randn(9, 3, 64, 64)
