
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv(x1)
        v1 = x1 + 3
        v2 = torch.clamp(v1, 0, 6)
        v3 = v2 / 6
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
