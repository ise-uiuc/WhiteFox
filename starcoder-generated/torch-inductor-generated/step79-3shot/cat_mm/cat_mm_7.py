
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, v1)
        v3 = torch.mm(v2, v2)
        v4 = torch.mm(v3, v3)
        v5 = torch.mm(v4, v4)
        return torch.cat([v2, v3, v4, v5], 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
