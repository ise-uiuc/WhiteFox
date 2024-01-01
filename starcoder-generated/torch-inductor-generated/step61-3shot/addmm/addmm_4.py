
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.init.zeros_(self.t1)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x1)
        v2 = v1 + x2
        return v2 ** 2
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
