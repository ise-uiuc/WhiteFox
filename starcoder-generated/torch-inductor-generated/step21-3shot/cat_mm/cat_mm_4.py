
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(v1, x2)
        v2 = torch.mm(x1, x2)
        v2 = torch.mm(v2, x2)
        v2 = torch.mm(v2, x2)
        v3 = torch.mm(x1, x2)
        v3 = torch.mm(v3, x2)
        v4 = torch.mm(x1, x2)
        v4 = torch.mm(v4, x2)
        return torch.cat([v1, v2, v3, v4], 1)
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(10, 1)
