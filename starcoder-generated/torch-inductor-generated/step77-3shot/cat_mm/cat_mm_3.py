
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        a = torch.mm(x, y)
        b = torch.mm(x, y)
        c = torch.mm(x, y)
        d = torch.mm(x, y)
        e = torch.mm(x, y)
        return torch.cat([c, b, d, a, e], 1)
# Inputs to the model
x = torch.randn(1, 1)
y = torch.randn(1, 2)
