
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = F.hardtanh(v2, min_val=0.0, max_val=6.0)
        v4 = torch.clamp(v3, 0, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        c1 = self.conv(x2)
        c2 = c1 + 3
        c3 = F.hardtanh(c2, min_val=0.0, max_val=6.0)
        c4 = torch.clamp(c3, 0, 6)
        c5 = c1 * c4
        c6 = c5 / 6
        return v6 + c6
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
