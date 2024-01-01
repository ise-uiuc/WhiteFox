
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        a1 = torch.tanh(x2)
        v1 = x2 + a1
        v2 = x1 + v1
        v3 = torch.sinh(v2)
        a2 = x3
        v4 = v3 + a2
        v5 = x2 + v4
        v6 = torch.sinh(v5)
        a3 = torch.tanh(x3)
        v7 = x3 + a3
        v8 = x1 + v7
        v9 = torch.sin(v8)
        a4 = x3
        v10 = v9 + a4
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
