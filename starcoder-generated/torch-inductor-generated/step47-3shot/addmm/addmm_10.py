
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, inp)
        v3 = torch.mm(v2, x3)
        v4 = v3 + x4
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 2)
x4 = torch.randn(3, 2)
inp = torch.randn(2, 2)
