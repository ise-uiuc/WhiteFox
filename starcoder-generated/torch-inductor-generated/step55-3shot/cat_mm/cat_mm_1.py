
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = torch.zeros(5, 5)
        v = v[(x1[0] + x1[1]).ge(3)]
        return torch.cat([v, v, v, v], 0)
# Inputs to the model
x1 = torch.randn(2, 5, dtype=torch.long)
