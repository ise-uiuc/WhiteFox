
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU6(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
