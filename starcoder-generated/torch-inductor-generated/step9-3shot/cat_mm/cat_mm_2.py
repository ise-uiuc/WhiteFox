
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        v4 = torch.mm(x1, x2)
        v5 = torch.mm(x1, x2)
        v6 = torch.mm(x1, x2)
        v7 = torch.mm(x1, x2)
        v8 = torch.cat([v7, v1, v2, v3, v4, v5, v6], 0)
        return v8
# Inputs to the model
x1 = torch.randn(7, 1)
x2 = torch.randn(1, 7)
