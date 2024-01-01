
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = a1 + 3
        a3 = torch.clamp(a2, 0, 6)
        a4 = a3 / 6
        return a4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
