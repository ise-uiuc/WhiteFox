
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v0 = x1.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v4 = x2.permute(0, 2, 1)
        v5 = x3.permute(0, 2, 1)
        v6 = x3.permute(0, 2, 1)
        v7 = torch.bmm(v3, v3)
        v8 = torch.bmm(v4, v4)
        v9 = v7.permute(0, 2, 1)
        return torch.matmul(v2, torch.bmm(v6, v0))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
