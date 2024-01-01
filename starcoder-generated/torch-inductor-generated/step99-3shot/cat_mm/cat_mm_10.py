
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x2, x1)
        for loopVar1 in range(10):
            v = torch.mm(x2, x1)
        v = torch.mm(x2, x1)
        v = torch.mm(x2, x1)
        return torch.cat([v, v], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
