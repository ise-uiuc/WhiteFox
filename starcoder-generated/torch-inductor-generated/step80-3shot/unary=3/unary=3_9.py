
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(228, 128, 17, stride=2, padding=8, dilation=1, groups=128)
        self.conv2 = torch.nn.Conv2d(128, 30, 15, stride=1, padding=7, dilation=1, groups=30)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 228, 32, 32)
