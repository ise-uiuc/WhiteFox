
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x2, x1)
        v4 = torch.mm(x2, x1)
        v5 = torch.mm(x1, x2)
        v6 = torch.mm(x1, x2)
        v7 = torch.mm(x2, x1)
        v8 = torch.mm(x2, x1)
        v9 = torch.mm(x1, x2)
        v10 = torch.mm(x1, x2)
        v11 = torch.mm(x2, x1)
        v12 = torch.mm(x2, x1)
        return torch.cat([v1, v1, v1, v1, v2, v2, v2, v2, v3, v3, v3, v3, v4, v4, v4, v4, v5, v5, v5, v5, v6, v6, v6, v6, v7, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v10, v10, v10, v10, v11, v11, v11, v11, v12, v12, v12, v12], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
