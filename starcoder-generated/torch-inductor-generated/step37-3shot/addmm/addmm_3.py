
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        v1 = torch.mm(x1, x2)
        v2 = x3 + v1
        v3 = v2 + x4
        v4 = v3 + inp
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
