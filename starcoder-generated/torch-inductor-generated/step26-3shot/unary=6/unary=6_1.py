
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.conv(x2)
        v7 = v6 + 3
        v8 = torch.clamp(v7, 0, 6)
        v9 = v6 * v8
        v10 = v9 / 6
        return v5 + v10
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
