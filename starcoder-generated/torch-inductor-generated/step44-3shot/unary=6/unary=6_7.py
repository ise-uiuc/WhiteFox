
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, x):
        v3 = self.conv2d(x)
        v6 = self.conv2d(v3)
        v1 = 3 + v3
        v2 = torch.clamp(v1, 0, 6)
        v5 = v2 * v6
        v4 = v5 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
