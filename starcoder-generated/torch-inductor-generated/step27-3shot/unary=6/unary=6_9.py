
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        a1 = self.conv(x1)
        a2 = a1 + 3
        a3 = torch.clamp(a2, 0, 6)
        a4 = a1 * a3
        a5 = a4 / 6
        a6 = self.conv(x2)
        a7 = a6 + 3
        a8 = torch.clamp(a7, 0, 6)
        a9 = a6 * a8
        a10 = a9 / 6
        return a5 + a10
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
