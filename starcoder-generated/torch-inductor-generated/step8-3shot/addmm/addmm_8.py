
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x2)
        v2 = v1 + x1
        v3 = torch.mm(v2, v1)
        return v3
# Inputs to the model
x1 = torch.randn(9, 5)
x2 = torch.randn(5, 2)
inp = torch.randn(10, 5)
