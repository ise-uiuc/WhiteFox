
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2.clamp_min
        v4 = v3(0)
        v5 = v4.clamp_max
        v6 = v5(6)
        v7 = v6.div
        v8 = v7(6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
