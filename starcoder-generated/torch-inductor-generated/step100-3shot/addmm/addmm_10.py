
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        y = torch.transpose(self.inp, 0, 1)
        v2 = v1 + y
        return v2 + inp
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
