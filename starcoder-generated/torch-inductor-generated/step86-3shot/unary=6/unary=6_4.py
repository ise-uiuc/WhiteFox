
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sum(v1, [0, 1, 2])
        v3 = v2 + 3
        v4 = torch.clamp(v3, 0, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
