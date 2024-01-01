
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v2 = v1
        if other is not None:
            v2 = other
            v3 = self.conv(x1)
            v4 = v3
            v2 = v4
        v5 = v1 + v2
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
