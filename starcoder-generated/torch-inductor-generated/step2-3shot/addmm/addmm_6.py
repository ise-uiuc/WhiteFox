
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v0 = x1 - x2
        inp = inp - 2*inp
        v1 = torch.addmm(inp, x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(6, 12)
x2 = torch.randn(12, 6)
inp = torch.randn(6, 6)
