
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, v1, v2):
        v3 = torch.mm(x1, x2)
        v4 = v3 + v1
        v5 = torch.mm(x1, x2)
        y = torch.transpose(self.inp, 0, 1)
        v6 = v5 + v4
        v7 = torch.mm(x1, x2)
        v8 = v7 + v6
        v9 = torch.mm(x1, x2)
        v10 = v9 + v8
        v11 = torch.mm(x1, x2)
        v12 = v11 + v10
        v13 = torch.mm(x1, x2)
        v14 = v13 + v12
        v15 = torch.mm(x1, x2)
        v16 = v15 + v14
        y = torch.transpose(v2, 0, 1)
        v18 = v16 + y
        return v18
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
v1 = torch.randn(3, 3, requires_grad=True)
v2 = torch.randn(3, 3, requires_grad=True)
