
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = torch.matmul(x1.permute(0, 2, 1), x2.permute(0, 2, 1))
        v1 = torch.bmm(x2.permute(0, 2, 1), x1.permute(0, 2, 1))
        v2 = torch.bmm(v1, v0)
        v3 = torch.bmm(v2, v1)
        v4 = torch.bmm(v3, v2)
        v5 = torch.bmm(v4, v3)
        v6 = torch.bmm(v5, v4)
        v7 = torch.bmm(v6, v5)
        return (v0, v1, v2, v3, v4, v5, v6, v7)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
