
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(10000):
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
        return torch.cat([v, v], 0)
# Inputs to the model
x1 = torch.randn((2**10, 2**10), dtype=torch.double)
x2 = torch.randn((2**10, 1), dtype=torch.double)
