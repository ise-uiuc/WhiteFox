
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, inp):
        v1 = torch.mm(inp, x1)
        v2 = torch.mm(inp, x2)
        v3 = torch.mm(inp, x3)
        v4 = torch.mm(inp, x4)
        v5 = torch.mm(inp, x5)
        v6 = torch.mm(inp, x6)
        v7 = torch.mm(inp, x7)
        v8 = torch.mm(inp, x8)
        v9 = torch.mm(inp, x9)
        v10 = torch.mm(x2, x10)
        v11 = torch.mm(x12, x11)
        v12 = torch.mm(x13, x14)
        return v9 + v10 + v11 + v12
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3)
x6 = torch.randn(3, 3)
x7 = torch.randn(3, 3)
x8 = torch.randn(3, 3)
x9 = torch.randn(3, 3)
x10 = torch.randn(3, 3)
x11 = torch.randn(3, 3)
x12 = torch.randn(3, 3)
x13 = torch.randn(3, 3)
x14 = torch.randn(3, 3)
inp = torch.randn(3, 3)
