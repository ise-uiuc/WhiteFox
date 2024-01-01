
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x1 + 3
        x4 = torch.clamp(x3, min=0, max=6)
        x5 = x4 / 6
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
