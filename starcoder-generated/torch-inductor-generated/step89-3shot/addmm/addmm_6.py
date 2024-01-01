
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, v0):
        v1 = x1 @ x2
        v2 = v1 + v0
        v3 = torch.mm(inp, inp)
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
v0 = torch.randn(3, 3)
