
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        def loop(t):
            v1 = torch.mm(x1, x2)
            return torch.cat([v1, v1], 1)
        t1 = loop(x1)
        t2 = loop(x2)
        return torch.cat([t1, t2, t1, t2, t1, t2, t1, t2, t1, t2], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
