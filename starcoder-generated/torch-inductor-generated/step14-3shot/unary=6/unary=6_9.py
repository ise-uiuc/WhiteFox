
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2)
        self.conv = torch.nn.Conv2d(3, 3, 4, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, 0, 6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 128, 128)
