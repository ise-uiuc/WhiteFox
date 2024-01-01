
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(4, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.add(v1, 3)
        v3 = self.conv(v2)
        v4 = 1 - v2
        v5 = torch.add(v3, v4)
        v6 = v5 + 2 * v1
        v7 = v6 * 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
