
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(3, 12, 64, 64)
