
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v1 * 0.5
        v5 = v3 + 1
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
