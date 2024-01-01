
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 2, stride=2, padding=0)
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = x1 + 3
        x3 = torch.clamp(x2, 0, 6)
        x4 = x1 * x3
        x5 = x4 / 6
        x6 = torch.max(x5, 1)[0]
        return x6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
