
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.hardtanh(v1, 3, 6)
        v3 = v2 + 3
        v4 = v3.clamp_min(0)
        v5 = v4.clamp_max(6)
        v6 = v5 / 6
        return v6
    def hardtanh(self, x, min_val=-1.0, max_val=1.0):
        return torch._C._nn.hardtanh(x, min_val, max_val)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
