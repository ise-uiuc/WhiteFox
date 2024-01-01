
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        b1 = self.conv(x1)
        b2 = b1 + 3
        b3 = torch.clamp(b2, 0, 6)
        b4 = b1 * b3
        b5 = b4 / 6
        return b5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
