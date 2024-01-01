
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, x1)
        v2 = torch.mm(inp, v1)
        v3 = torch.mm(x2, v1)
        v4 = torch.mm(v2, v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
