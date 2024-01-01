
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.randn(4, 2, 3)
        v2 = torch.randn(4, 2, 5)
        v3 = torch.randn(4, 2, 5)
        v4 = torch.bmm(v1, v1)
        v5 = torch.matmul(x2, x2)
        v6 = torch.bmm(v2, v2)
        v7 = torch.matmul(v6, v6)
        v8 = torch.bmm(v4, v2)
        return (v1, v2, v3, v4, v5, v6, v7, v8)
# Inputs to the model
x1 = torch.randn(4, 2, 2)
x2 = torch.randn(4, 2, 2)
