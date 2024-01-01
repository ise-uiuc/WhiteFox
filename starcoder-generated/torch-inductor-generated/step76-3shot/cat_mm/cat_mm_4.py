
class Model(torch.nn.Module):
    def __init__(self, loopVar1, loopVar2):
        super().__init__()
        self.loopVar1 = loopVar1
        self.loopVar2 = loopVar2
    def forward(self, x1, x2):
        v = []
        for loopVar3 in range(self.loopVar1):
            t1 = torch.mm(x1, x2)
            for loopVar4 in range(self.loopVar2):
                v.append(torch.mm(t1, x2))
        return torch.cat(v, 1)
loopVar1 = 3
loopVar2 = 2
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
