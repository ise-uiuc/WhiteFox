
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, z)
        v3 = torch.mm(y, z)
        return torch.cat([v1, v2, v3, v3, v2], 1)
# Inputs to the model
x = torch.randn(3, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
