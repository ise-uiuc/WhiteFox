
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, groups=3)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0.0, max=6.0)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
