
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        t1 = v1 * v2 * v3
        v4 = torch.mm(x1, x2)
        v5 = torch.mm(x1, x2)
        v6 = torch.mm(x1, x2)
        t2 = v4 * v5 * v6
        v7 = torch.mm(x1, x2)
        v8 = torch.mm(x1, x2)
        v9 = torch.mm(x1, x2)
        t3 = v7 * v8 * v9
        v10 = torch.mm(x1, x2)
        v11 = torch.mm(x1, x2)
        v12 = torch.mm(x1, x2)
        t4 = v10 * v11 * v12
        t5 = torch.cat([t1, t2], 1)
        t6 = torch.cat([t3, t4], 1)
        return v1 * t5 * v1 * t6
# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 2)
x3 = torch.randn(2, 2)
