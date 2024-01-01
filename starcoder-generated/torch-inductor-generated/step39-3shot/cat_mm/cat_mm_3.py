
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = torch.mm(x2, x3)
        v = torch.cat([v, v], 2)
        v = torch.mm(x1, v)
        return torch.cat([v, v, v], 0)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
