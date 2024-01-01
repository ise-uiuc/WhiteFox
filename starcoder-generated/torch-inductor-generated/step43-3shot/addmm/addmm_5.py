
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        m1 = torch.mm(x1, x1)
        b1 = torch.add(m1, x1 + x2)
        out = torch.add(b1, inp)
        return out
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, 3, requires_grad=True)
