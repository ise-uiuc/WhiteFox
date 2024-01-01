
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, y)
        t1 = torch.cat([v1, v1], 0)
        t2 = torch.cat([t1, t1, t1, t1], 2)
        t3 = torch.cat([t1, t1, t1], 1)
        t4 = torch.cat([t3, t3, t3, t3], 1)
        return torch.cat([t4, t4], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(3, 2)
