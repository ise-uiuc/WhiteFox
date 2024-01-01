
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3, requires_grad=True)
        self.inp2 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, inp=None):
        if inp is not None:
            inp = self.inp2
        else:
            inp = x2
        t1 = torch.mm(x1, inp)
        t2 = torch.mm(x1, t1)
        t3 = t1 + x1
        return t3, t2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
