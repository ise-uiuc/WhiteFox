
class Model(torch.nn.Module):
    def __init__(self, loopVar):
        super().__init__()
        self.loopVar = loopVar
    def forward(self, x1, x2):
        v = []
        for loopVar3 in range(self.loopVar):
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x2, x2))
        return torch.cat(v, 1)
loopVar = 3
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
