
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.hardtanh = torch.nn.Hardtanh(0, 6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = self.hardtanh(v2)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
