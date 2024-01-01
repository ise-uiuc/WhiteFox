
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=(0, 0))
    def forward(self, x1):
        z1 = self.conv(x1)
        z2 = z1 + 3
        z3 = z2.clamp(min=0, max=6)
        z4 = z3.div(6)
        return z4
# Inputs to the model
a1 = torch.randn(1, 3, 500, 500)
