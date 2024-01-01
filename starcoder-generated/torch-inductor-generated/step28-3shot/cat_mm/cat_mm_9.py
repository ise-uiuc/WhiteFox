
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c):
        v1 = torch.cat([a], 1)
        v2 = b + a
        v3 = torch.cat([c, v2, b, b, a, a], 1)
        return torch.cat([v1, v3, v2, v3, v2, v3, v1], 0)
# Inputs to the model
a = torch.mm(torch.randn(2, 3), torch.randn(3, 4))
b = torch.mm(torch.randn(2, 2), torch.randn(2, 4))
c = torch.mm(torch.randn(2, 5), torch.randn(5, 4))

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c):
        v1 = torch.cat([a], 1)
        v2 = b + a
        v3 = torch.cat([a, b, c, v2, a, b, c, b, c, v2, b, c, a, v2, c, a, b, c, b, c, a], 1)
        return torch.cat([v1, v3, v2, v3, v2, v3, v1], 0)
# Inputs to the model
a = torch.mm(torch.randn(2, 3), torch.randn(3, 4))
b = torch.mm(torch.randn(2, 2), torch.randn(2, 4))
c = torch.mm(torch.randn(2, 5), torch.randn(5, 4))
