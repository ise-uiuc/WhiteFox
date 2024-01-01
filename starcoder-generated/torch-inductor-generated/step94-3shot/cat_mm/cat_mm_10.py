
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        v4 = torch.mm(x1, x2)
        v5 = torch.mm(x1, x2)
        return torch.cat([v1, v3, v3, v2, v4, v4, v5, v4, v2], 1)
# Inputs to the model
x1 = torch.randn(4, 5)
x2 = torch.randn(5, 4)
