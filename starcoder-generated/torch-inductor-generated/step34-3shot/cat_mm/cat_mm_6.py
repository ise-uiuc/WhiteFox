
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        v = torch.cat([torch.mm(x1, x2), torch.mm(x3, x4), torch.mm(x5, x1), torch.mm(x1, x2)], 1)
        return torch.cat([v, v], 1)
# Inputs to the model
x1 = torch.randn(4, 2)
x2 = torch.randn(2, 4)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 4)
x5 = torch.randn(2, 2)
