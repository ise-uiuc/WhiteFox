
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        v3 = v1 * 3
        v4 = v2 + v3
        v5 = v4 - 3
        v6 = torch.clamp(v5, min=0, max=6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
