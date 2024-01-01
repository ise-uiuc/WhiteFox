
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 5, stride=1, padding=2)
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        return v7.permute(2, 3, 1, 0).unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
