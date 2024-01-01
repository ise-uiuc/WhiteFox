
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 2, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.clamp_min(v1, 0)
        v3 = 6.0
        v4 = torch.clamp_max(v2, 8)
        v5 = v4 + v3
        v6 = v3 * v5
        v7 = v6 / 3.0
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
