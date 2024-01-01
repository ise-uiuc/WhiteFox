
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        z1 = torch.cat([x1, x2], 1)
        z2 = torch.abs(z1)
        z3 = z2 * 5
        z4 = z3 + 6
        z5 = torch.clamp_min(z4, 0)
        z6 = torch.clamp_max(z5, 6)
        z7 = z1 + z6
        z8 = z7 / 6
        return z8
# Inputs to the model
x1 = torch.randn(2, 8, 28, 28)
x2 = torch.randn(2, 8, 28, 28)
