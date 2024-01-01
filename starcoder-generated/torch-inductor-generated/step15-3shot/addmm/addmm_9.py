
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, torch.mm(inp, inp.T))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
inp = torch.randn(1, 1)
