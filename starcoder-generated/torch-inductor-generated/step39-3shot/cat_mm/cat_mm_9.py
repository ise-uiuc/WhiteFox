
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        if x1.size()[1] < x2.size()[0]:
            for loopVar1 in range(x1.size()[1]):
                v = torch.mm(x1, x2)
            for loopVar1 in range(x1.size()[1]):
                v = torch.mm(x1, x2)
        for loopVar1 in range(x2.size()[1]):
            v = torch.mm(x1, x2)
        return torch.cat([v, v], 1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(2, 2)
