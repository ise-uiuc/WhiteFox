
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 16, 2, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x2 = torch.randn(1, 1, 23, 23)
