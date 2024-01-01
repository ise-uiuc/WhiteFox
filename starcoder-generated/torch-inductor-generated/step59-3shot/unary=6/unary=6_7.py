
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.hardtanh = torch.nn.Hardtanh(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.bn(v5)
        t1 = self.hardtanh(v6)
        v7 = torch.clamp(t1, min=0, max=6)
        v8 = v5 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
