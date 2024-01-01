
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 100, 3, stride=2, padding=1)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.pool(v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
