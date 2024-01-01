
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        z1 = torch.add(x1, 3)
        z2 = self.conv(z1)
        z3 = z2.clamp(min=1, max=3)
        z4 = z3.div(3)
        return z4
# Inputs to the model
a1 = torch.randn(1, 3, 64, 64)
