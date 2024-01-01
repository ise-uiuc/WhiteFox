
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v2 = self.conv(x1) + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp(max=6)
        v6 = v4 / 6
        return v6
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
