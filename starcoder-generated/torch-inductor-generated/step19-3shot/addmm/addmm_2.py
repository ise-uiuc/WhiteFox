
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, inp)
        t1 = v1 + inp
        v2 = torch.mm(x1, x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(6, 1)
inp = torch.randn(6, 6)
