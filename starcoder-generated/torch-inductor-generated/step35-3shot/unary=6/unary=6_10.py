
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.hardtanh = torch.nn.Hardtanh(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        t1 = self.hardtanh(v7)
        v8 = torch.clamp_min(t1, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v6 * v9
        v11 = v10 / 6
        return v11
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
