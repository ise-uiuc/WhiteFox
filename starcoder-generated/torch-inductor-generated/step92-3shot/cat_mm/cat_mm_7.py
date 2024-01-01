
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x1)
        v3 = torch.mm(x1, x3)
        v4 = torch.mm(x1, x3)
        v5 = torch.mm(x1, x3)
        v6 = torch.mm(x2, x3)
        v7 = torch.mm(x2, x3)
        v = torch.cat([v1, v2, v3, v4, v5, v6, v7], 1)
        return v
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
