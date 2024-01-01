
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3, requires_grad=True)
        self.inp2 = torch.randn(3, 3, requires_grad=True)
        self.inp3 = torch.randn(1, 3, requires_grad=True)
    def forward(self, x1):
        t1 = torch.mm(x1, x1)
        t2 = torch.mm(x1, x1)
        t3 = t2 + self.inp3
        return t1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
