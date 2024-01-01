
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1 + x2
        v3 = v1 + x3
        v4 = v3 + inp
        v5 = torch.mm(v4, x2)
        v6 = x6 + x4
        v7 = torch.mm(x5, v6)
        v8 = v5 + v7
        return v8
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3, requires_grad=False)
x5 = torch.randn(3, 3, requires_grad=True)
x6 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
