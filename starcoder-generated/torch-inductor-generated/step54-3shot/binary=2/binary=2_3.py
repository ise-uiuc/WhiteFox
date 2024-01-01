
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x, x1, x2, x3, x4):
        v1 = self.conv(x)
        v2 = v1 - x2
        v3 = self.conv1(x1)
        v4 = v3 + x4
        v5 = v2 - v4
        # t105 = clip(t104, min=float('-inf'), max=3.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
x5 = torch.randn(1, 3, 64, 64)
