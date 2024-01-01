
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
    def forward(self, x2, x3, x4, x5):
        v1 = self.conv(x2)
        v2 = v1 - 1.0
        v3 = torch.sinh(x3)
        v4 = v2 * v3
        v5 = torch.nn.functional.hardtanh(x4)
        v6 = v4 - v5
        v7 = v6 + x5
        return v7
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1)
x4 = torch.randn(1)
x5 = torch.randn(1, 8, 63, 63)
