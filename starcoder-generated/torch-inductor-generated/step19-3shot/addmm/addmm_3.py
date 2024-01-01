
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward( self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1.mm(x3)
        return v1.mm(inp + v2) + torch.mm(x1, inp)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 1)
x3 = torch.randn(1, 3)
inp = torch.randn(2, 3)
