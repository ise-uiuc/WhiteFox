
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, v0, v1):
        v2 = x5 * x2
        v3 = v2 + v1
        v4 = torch.mm(x1, x4)
        v5 = v4 + v3
        v6 = torch.mm(x2, x5)
        v7 = v5 + v6
        v8 = torch.mm(x3, v1)
        v9 = v7 * v8
        return v3 + v0 + v9
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3, requires_grad=True)
x5 = torch.randn(3, 3, requires_grad=True)
v0 = torch.randn(3, 3)
v1 = torch.randn(3, 3)
