
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 8, stride=1, padding=2)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        return self.relu(self.bn(v5))
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
