
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = torch.randn(1, 3, 64, 64)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v3 = v3 + 3
        v3 = torch.clamp_min(v3, 0)
        v3 = torch.clamp_max(v3, 6)
        v3 = v3 * 2
        v4 = v3 * v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
