
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x3)
        v3 = torch.mm(x1, x4)
        v4 = torch.mm(x1, x5)
        v5 = torch.mm(x1, x6)
        v6 = v1 + v2 + v3 + v4 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
x4 = torch.randn(1, 1)
x5 = torch.randn(1, 1)
x6 = torch.randn(1, 1)
