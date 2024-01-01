
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v4 = torch.cat([x1, x2, x3], -1)
        v1 = v4.reshape(-1, 3, 3)
        v5 = v1.permute(0, 2, 1)
        v6 = torch.ones_like(v5)
        v7 = torch.matmul(v5.permute(0, 2, 1), v6.permute(0, 2, 1))
        v8 = v7.permute(0, 2, 1)
        v2 = torch.ones_like(v4)
        v9 = v2.reshape(-1, 3, 3)
        v10 = v9.permute(0, 2, 1)
        v3 = torch.ones_like(v4)
        v11 = v3.reshape(-1, 3, 3)
        v12 = v11.permute(0, 2, 1)
        v13 = torch.mm(v12, v10)
        return v8 + v13
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 4, 3)
x3 = torch.randn(1, 6, 3)
