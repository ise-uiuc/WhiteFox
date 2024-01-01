
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=3, padding=15)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = v2 * 0.1
        v6 = v5 + v4
        v7 = torch.clamp_max(v6, 6)
        v8 = 2 * v5
        v9 = v8 - v6
        v10 = v5 / 6
        v11 = v9 / 5
        return v11
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
