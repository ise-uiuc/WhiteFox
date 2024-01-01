
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 - v3
        v5 = v4 / 6
        v7 = self.conv2(v1)
        v8 = self.conv2(v7 + 1)
        v9 = torch.clamp(3, 0, 6)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
