
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 + 3
        x4 = torch.clamp_min(x3, 0)
        x5 = torch.clamp_max(x3, 6)
        x6 = x2 * x5
        x7 = x6 / 6
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
