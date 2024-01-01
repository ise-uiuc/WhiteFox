
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv(x1)
        v2 = x1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
