
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x2)
        v2 = v1 + x1
        return v2.T
# Inputs to the model
x1 = torch.randn(100, 100)
x2 = torch.randn(0, 0)
inp = torch.randn(100, 0)
