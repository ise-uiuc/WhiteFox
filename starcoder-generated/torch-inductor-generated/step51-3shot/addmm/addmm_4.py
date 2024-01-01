
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + x2
        v3 = v2 + inp
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, requires_grad=True)
x2 = torch.randn(1, 1, requires_grad=True)
inp = torch.randn(1, 1, requires_grad=True)
