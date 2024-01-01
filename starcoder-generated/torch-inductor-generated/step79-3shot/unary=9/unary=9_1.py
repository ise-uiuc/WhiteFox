
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = 3 + v1
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4 / 6
        v6 = self.conv2(v5) + 3
        v7 = v6.clamp_min(0)
        v8 = v7.clamp_max(6)
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
