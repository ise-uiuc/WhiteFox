
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 2, stride=2, padding=23)
        self.conv2 = torch.nn.Conv2d(16, 8, 2, stride=2, padding=61)
        self.conv3 = torch.nn.Conv2d(8, 12, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2(self.conv(x1))
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 112, 240)
