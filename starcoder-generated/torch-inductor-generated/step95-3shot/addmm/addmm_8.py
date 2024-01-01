
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        z1 = torch.mm(x1, inp)
        z2 = x3 * self.add(x4, inp)
        z3 = z1 + x4
        z4 = (z1 * z2) + x4 + x3
        return z3, z4
    def add(self, x5, x6):
        return x5 + x6
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3)
x4 = torch.randn(3, requires_grad=True)
inp = torch.randn(3, 3)
