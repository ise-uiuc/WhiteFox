
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x1)
        v3 = v1
        v4 = v1
        v5 = torch.mm(x2, x2)
        v6 = v5
        v7 = v5
        v8 = torch.mm(x3, x3)
        v9 = v8
        v10 = v8
        v11 = torch.mm(x4, x4)
        v12 = v11
        v13 = v11
        v2 = v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13
        return v2
# Inputs to the model
x1 = torch.randn(1, 65)
x2 = torch.randn(65, 5)
x3 = torch.randn(1, 65)
x4 = torch.randn(65, 5)
