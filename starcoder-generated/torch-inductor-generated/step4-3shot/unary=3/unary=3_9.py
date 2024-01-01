
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
