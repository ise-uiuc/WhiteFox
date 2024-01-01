
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        t1 = torch.cat([v1, v1], 1)
        t2 = torch.cat([t1, t1], 1)
        t3 = torch.cat([t2, t2], 1)
        t4 = torch.cat([t3, t3], 1)
        return torch.cat([t4, t4], 1)
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
