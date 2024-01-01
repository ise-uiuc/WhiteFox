
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1 + 0.1    # v1+0.1: [1, 3, 64, 64] + [1],   [1, 3, 64, 64]
        v4 = torch.relu(v3)
        return v4
x1 = torch.randn(1, 3, 64, 64)   # [1, 3, 64, 64]
