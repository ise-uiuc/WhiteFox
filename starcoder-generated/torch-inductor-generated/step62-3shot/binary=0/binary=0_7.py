
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        other1 = other.repeat(1, 1, 32, 32)
        other2 = other.repeat(1, 1, 16, 16)
        v2 = v1 + other1
        v3 = v2 + other2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
