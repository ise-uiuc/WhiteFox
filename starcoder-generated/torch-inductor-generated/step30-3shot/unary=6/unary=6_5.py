
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        d6 = 3
        v1 = self.conv(x1)
        v2 = v1 + d6
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 384, 384)
