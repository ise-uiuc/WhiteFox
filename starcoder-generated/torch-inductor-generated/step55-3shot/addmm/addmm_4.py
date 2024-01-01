
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        v3 = v2 - x3
        v4 = v3 * x4
        return v1 + v2 + v3 + v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
inp = torch.randn(3, 3)
