
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = torch.randn(3, 3)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, x1)
        v2 = v1 + inp
        x1 = x2 + v1
        v1 = v1 + x2
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
