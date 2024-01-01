.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.conv2(v6)
        v8 = v7 + 3
        v9 = torch.clamp_min(v8, 0)
        v10 = torch.clamp_max(v9, 6)
        v11 = v7 * v10
        v12 = v11 / 6
        return v12
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
