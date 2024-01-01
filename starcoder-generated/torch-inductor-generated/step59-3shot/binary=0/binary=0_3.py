
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 1, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v5 = (v2 + x3) + x4
        v6 = torch.cat([v2, x5], 1)
        v9 = (v5 + v6) * x2
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
x4 = torch.randn(1, 1, 2, 2)
x5 = torch.randn(1, 1)
