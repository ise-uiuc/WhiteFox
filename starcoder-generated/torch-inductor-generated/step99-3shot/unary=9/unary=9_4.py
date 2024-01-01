
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        i0 = 3
        v2 = v1 + i0
        v3 = torch.clamp(v2, -2147483648, 2147483647)
        v4 = v3.div(-2147483647)
        v5 = torch.clamp(v4, -2147483648, 2147483647)
        v6 = v5 / 2147483647
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
