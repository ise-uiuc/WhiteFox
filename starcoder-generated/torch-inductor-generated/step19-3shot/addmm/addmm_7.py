
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        t1 = torch.mm(inp, inp)
        v1 = t1 + x1
        v2 = torch.mm(v1, x2)
        return v2
# Inputs to the model
inp = torch.randn(4, 3)
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 2)
