
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
    def forward(self, x1):
        f1 = self.conv(x1)
        f2 = 3 + f1
        f3 = torch.clamp(f2, 0, 6)
        f4 = f1 * f3
        f5 = f4 / 6
        return f5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
