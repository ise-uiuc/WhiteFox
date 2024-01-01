
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(5, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v1 / v2 * v3 + torch.tanh(v4) - v5 + v6
        v8 = torch.abs(v7)
        v9 = self.conv3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 16, 14)
