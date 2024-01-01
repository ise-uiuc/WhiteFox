
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        v1 = torch.mm(x, x)
        t2 = torch.cat([t1, t1, t1, t1], 1)
        t3 = torch.mm(x, x)
        v2 = torch.mm(x, x)
        return torch.cat([v1, t2, v2, v1], 1)
# Inputs to the model
x = torch.randn(2, 2)
