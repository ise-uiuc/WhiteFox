
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x1)
        v3 = torch.mm(x2, x1)
        v4 = torch.mm(x1, x2)
        v5 = torch.mm(x2, x1)
        v6 = torch.mm(x1, x2)
        v7 = torch.mm(x2, x1)
        v8 = torch.mm(x1, x2)
        v9 = torch.mm(x1, x1)
        v10 = torch.mm(x1, x2)
        v11 = torch.mm(x1, x2)
        v12 = torch.mm(x1, x2)
        v13 = torch.mm(x1, x2)
        v14 = torch.mm(x2, x1)
        v15 = torch.mm(x2, x1)
        v16 = torch.mm(x2, x1)
        v17 = torch.mm(x2, x1)
        v18 = torch.mm(x2, x1)
        v19 = torch.mm(x2, x1)
        v20 = torch.mm(x2, x1)
        v21 = torch.mm(x1, x2)
        v22 = torch.mm(x2, x1)
        v23 = torch.mm(x2, x1)
        return torch.cat([v1, v1, v1, v1, v2, v2, v2, v2, v3, v3, v3, v3, v4, v4, v4, v4, v5, v5, v5, v5, v6, v6, v6, v6, v7, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v10, v10, v10, v10, v11, v11, v11, v11, v12, v12, v12, v12, v13, v13, v13, v13, v14, v14, v14, v14, v15, v15, v15, v15, v16, v16, v16, v16, v17, v17, v17, v17, v18, v18, v18, v18, v19, v19, v19, v19, v20, v20, v20, v20, v21, v21, v21, v21, v22, v22, v22, v22, v23, v23, v23, v23], -1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
