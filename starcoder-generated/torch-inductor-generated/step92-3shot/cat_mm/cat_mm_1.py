
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(7):
            v = torch.mm(x1, x2)
        for loopVar1 in range(6):
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
        return torch.cat([v, v, v], 0)
# Inputs to the model
x1 = torch.randn(5, 7)
x2 = torch.randn(7, 5)
