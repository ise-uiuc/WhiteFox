
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        v4 = torch.mm(v1, v2)
        v5 = torch.mm(v2, v3)
        v6 = torch.mm(v3, v4)
        return torch.cat([torch.mm(v1, v2), torch.mm(v2, v3), torch.mm(v3, v4), torch.mm(v1, v2), torch.mm(v2, v3), torch.mm(v3, v4)], 1)
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(4, 1)
