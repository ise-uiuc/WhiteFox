
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, inp)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(2, 7)
inp = torch.randn(3, 2)
