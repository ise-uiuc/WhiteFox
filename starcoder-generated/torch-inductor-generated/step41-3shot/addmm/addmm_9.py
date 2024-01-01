
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(30, 40, requires_grad=True)
x2 = torch.randn(30, requires_grad=True)
inp = torch.randn(40, requires_grad=True)
