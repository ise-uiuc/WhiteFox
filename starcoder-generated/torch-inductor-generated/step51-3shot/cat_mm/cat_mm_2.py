
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v1, v1, v1], 1)
        v2 = torch.mm(x1, x3)
        t2 = torch.cat([v2, v2], 1)
        return torch.cat([t1, t1, t1, t2, t2, t2], 0)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
x3 = torch.randn(2, 3)
