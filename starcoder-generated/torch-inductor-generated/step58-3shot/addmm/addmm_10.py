
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.nn.Dropout2d(p=0.2)(x1)
        v2 = torch.mm(x1, x2)
        v3 = v2 + inp
        return v3
# Inputs to the model
x1 = torch.randn(111, 111, requires_grad=True)
x2 = torch.randn(111, 111, requires_grad=True)
inp = torch.randn(111, 111, requires_grad=True)
