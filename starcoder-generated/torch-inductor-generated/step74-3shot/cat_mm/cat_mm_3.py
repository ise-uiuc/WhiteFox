
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
        v2 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
        v3 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
        v4 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
        v5 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
        return torch.cat([v1, v2, v3, v4, v5, v1], 0)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 2)
