
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        v3 = torch.mm(x, y)
        v4 = torch.mm(x, y)
        v5 = torch.mm(x, y)
        v6 = torch.mm(x, y)
        v7 = torch.mm(x, y)
        v8 = torch.mm(x, y)
        v9 = torch.mm(x, y)
        v10 = torch.mm(x, y)
        v11 = torch.mm(x, y)
        v12 = torch.mm(x, y)
        v13 = torch.mm(x, y)
        v14 = torch.mm(x, y)
        v15 = torch.mm(x, y)
        return torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15], 1)
# Inputs to the model
x = torch.randn(3, 2)
y = torch.randn(2, 2)
