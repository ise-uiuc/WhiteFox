
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, torch.zeros_like(x2))
        v3 = torch.mm(x1, torch.zeros_like(x2))
        v4 = torch.mm(x1, torch.zeros_like(x2))
        v5 = torch.mm(x1, torch.zeros_like(x2))
        v6 = torch.mm(x1, v5)
        v7 = torch.mm(x2, v6)
        v8 = torch.mm(v7, v6)
        return torch.cat([v1, v2, v3, v4, v5, v6, v7, v8], 0)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
