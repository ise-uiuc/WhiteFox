
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1.5470053837925151
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
