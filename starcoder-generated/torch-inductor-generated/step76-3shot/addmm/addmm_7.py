
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        t1 = x1 + inp
        t2 = x2 + inp
        t3 = torch.mm(t1, t2)
        return t2, t3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
