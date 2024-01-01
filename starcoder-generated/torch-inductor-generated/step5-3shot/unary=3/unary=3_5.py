
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential()
        self.conv1.add_module('a', torch.nn.Conv2d(1, 8, 1, stride=1, padding=1))
    def forward(self, x1):
        v1 = self.conv1.a(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
