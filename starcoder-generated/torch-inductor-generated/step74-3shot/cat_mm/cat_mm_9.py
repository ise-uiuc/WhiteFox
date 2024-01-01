
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar in range(6):
            v.append(torch.mm(x1, x2))
        t1 = torch.cat(v, 1)
        for loopVar2 in range(7):
            v.append(torch.mm(x1, x2))
        v.append(torch.mm(x1, x2))
        t2 = torch.cat(v, 1)
        return torch.cat([t1, t2], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
