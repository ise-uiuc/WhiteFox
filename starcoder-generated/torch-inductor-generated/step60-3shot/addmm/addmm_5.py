
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3, requires_grad=True)
        self.inp2 = torch.randn(3, 3, requires_grad=True)
        self.inp3 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2=None):
        if (x2 is not None):
            inp = self.inp3
        else:
            inp = x1
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(x1, x1)
        v3 = v1 + inp
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
