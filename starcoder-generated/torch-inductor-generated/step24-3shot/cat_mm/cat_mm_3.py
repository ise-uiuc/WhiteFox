
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x1)
        t1 = torch.cat([v1, v1, v1, v1], 1)
        t2 = torch.cat([t1, t1, t1], 1)
        t3 = torch.cat([t2, v1], 1)
        return torch.cat([t3, v2], 1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 2)
