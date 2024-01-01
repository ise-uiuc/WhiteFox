
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3, x4):
        v0 = x1.transpose(-2, -1)
        v1 = x2.transpose(-2, -1)
        v2 = x3.transpose(-2, -1)
        v3 = x4.transpose(-2, -1)
        v4 = torch.matmul(v2, v3)
        v5 = torch.matmul(v1, v4)
        v6 = torch.matmul(v0, v1)
        v7 = torch.matmul(v6, v0)
        v8 = torch.matmul(v7, v6)
        v9 = torch.matmul(v5, v0)
        v10 = torch.matmul(v0, v9)
        v11 = torch.matmul(v10, v7)
        v12 = torch.matmul(v1, v7)
        v13 = torch.matmul(v5, v10)
        v14 = torch.matmul(v6, v9)
        v15 = torch.matmul(v1, v14)
        v16 = torch.matmul(v15, v0)[..., 0]
        v17 = torch.cat((v13, v2), 1)
        return v17
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
