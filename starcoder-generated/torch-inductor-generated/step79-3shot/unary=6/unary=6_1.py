
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1) + 3
        v2 = torch.clamp(v1, 0, 6)
        v3 = torch.clamp(v2 + 3, 0, 6)
        v4 = v1 * v2
        v5 = v3 / 6
        v6 = v1 / 6
        v7 = v5 + v6 + 3
        v8 = v7 / 3
        return v8
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
