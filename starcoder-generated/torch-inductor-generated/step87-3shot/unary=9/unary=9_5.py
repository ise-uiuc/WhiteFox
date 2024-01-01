
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = torch.clamp(v2, 0, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 6, 64, 64)
