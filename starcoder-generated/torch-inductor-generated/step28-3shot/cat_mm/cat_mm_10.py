
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        v3 = torch.mm(x, y)
        v4 = torch.mm(x, y)
        v5 = torch.mm(x, y)
        v6 = torch.mm(x, y)
        v7 = torch.mm(x, y)
        v8 = torch.mm(x, y)
        return torch.cat([v1, v2, v3, v4, v5, v6, v7, v8], 1)
# Inputs to the model
x = torch.randn(2, 4)
y = torch.randn(3, 4)
