
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        x2 = v1 + 3
        x3 = torch.clamp_min(x2, 0)
        x4 = torch.clamp_max(x3, 6)
        x5 = v1 * x4
        x6 = x5 / 6
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
