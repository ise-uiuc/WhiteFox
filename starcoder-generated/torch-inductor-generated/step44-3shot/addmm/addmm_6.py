
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        t0 = torch.mm(inp, inp)
        t1 = x1 * t0
        t2 = t1 + x1
        v1 = torch.mm(t2, inp)
        t3 = v1 + x2
        t4 = torch.mm(t3, t3)
        t5 = x4 + t4
        v2 = torch.mm(x3, t5)
        t6 = x2 * v2
        t7 = t6 + v2
        return t7
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
