
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x3)
        v2 = torch.mm(x2, v1)
        v3 = v2 + inp
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3)
inp = torch.randn(3, 3, 3, requires_grad=True)
