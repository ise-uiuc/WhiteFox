
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mul(x1, x2)
        v2 = torch.mul(inp, inp)
        v3 = torch.mm(v1, v2)
        return v3 + inp
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
