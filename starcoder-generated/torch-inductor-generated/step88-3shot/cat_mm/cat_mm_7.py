
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        result = torch.cat([v, v])
        for loopVar1 in range(100):
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
            result = torch.cat([v, v, v], 1)
            for loopVar2 in range(100):
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                v = torch.mm(x1, x2)
                result = torch.cat([v, v], 1)
        return result
# Inputs to the model
x1 = torch.randn(1, 400)
x2 = torch.randn(1, 400)
