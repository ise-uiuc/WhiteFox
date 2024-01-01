
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v2 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v3 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v4 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v5 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v6 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        return torch.cat([v1, v2, v3, v4, v5, v6], 1)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
