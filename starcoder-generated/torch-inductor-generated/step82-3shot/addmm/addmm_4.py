
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, y, inp):
        y = torch.mm(y, inp + x5)
        v1 = torch.mm(x1, inp)
        v2 = torch.mm(y + v1, x2) + x3
        v3 = torch.mm(v1, x4)
        v4 = torch.mm(v2 + x5, y)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3)
y = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
