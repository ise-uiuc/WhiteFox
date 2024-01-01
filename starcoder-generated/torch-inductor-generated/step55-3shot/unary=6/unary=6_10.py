
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 3, stride=2, padding=5)
    def forward(self, x1, x2):
        b1 = self.conv(x1)
        b2 = self.conv(x2)
        b3 = torch.add(b1, b2)
        b4 = torch.clamp_min(b3, 0)
        b5 = torch.clamp_max(b4, 6)
        b6 = b1 * b5
        b7 = b6 / 6
        return b3 + b7
# Inputs to the model
x1 = torch.randn(4, 3, 28, 28)
x2 = torch.randn(4, 3, 28, 28)
