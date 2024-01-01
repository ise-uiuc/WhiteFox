
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, a1, a2, a3, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        v3 = torch.mm(a1, a2)
        v2 = v2 + v3
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
a1 = torch.randn(3, 3)
a2 = torch.randn(3, 3)
a3 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
