
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 7, stride=2, padding=(2, 3))
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 3, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, 0, 6)
        v6 = v3 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
