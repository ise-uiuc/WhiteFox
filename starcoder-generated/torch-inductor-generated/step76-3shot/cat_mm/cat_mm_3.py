
class Model(torch.nn.Module):
    def __init__(self, loopVar):
        super().__init__()
        self.loopVar = loopVar
    def forward(self, x1, x2):
        v = []
        for loopVar3 in range(self.loopVar):
            v.append(torch.mm(x1, x2))
        return torch.cat(v, 2)
loopVar = 1
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 1, 1)
