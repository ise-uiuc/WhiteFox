
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        v1 = torch.mm(x, y)
        v2 = torch.mm(y, z)
        v4 = torch.mm(z, x)
        v6 = torch.mm(x, y)
        return torch.cat([v1, v1, v1, v2, v2, v4], 1)
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
