
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, inp)
        v2 = torch.mm(v1, inp)
        v3 = torch.mm(v1, v1)
        v4 = torch.mm(v2, v1)
        v5 = torch.mm(v2, v2)
        v6 = torch.mm(v4, v2)
        v7 = torch.mm(v6, v3)
        v8 = torch.mm(v6, v4)
        v9 = torch.mm(v6, v5)
        v10 = torch.mm(v8, v5)
        v11 = torch.mm(inp, v10)
        v12 = torch.mm(v10 + v11, v3)
        return v7 + v8 + v9 + v12
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
