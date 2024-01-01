
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1
        v2 = x1
        v3 = x2.permute(0, 2, 1)
        v4 = torch.matmul(v3, v1)
        v5 = v1.permute(0, 2, 1)
        v6 = x2
        v7 = torch.matmul(v5, v2)
        v8 = v7.permute(0, 2, 1)
        v9 = torch.matmul(v4, v6)
        v10 = x2
        v11 = torch.matmul(v9, v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
