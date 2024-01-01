
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=2, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x1 + 3
        x4 = torch.clamp(x1, 0, 6)
        x5 = x3 * x4
        x6 = x5 / 6
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
