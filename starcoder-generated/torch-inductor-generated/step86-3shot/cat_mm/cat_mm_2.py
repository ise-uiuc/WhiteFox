
class Model(torch.nn.Module):
    def __init__(self, loopVar):
        super().__init__()
        self.loopVar = loopVar
    def forward(self, x1, x2):
        v = []
        for loopVar3 in range(self.loopVar):
            v.append(x1)
            v.append(x2)
        return torch.cat(v, 0)
loopVar = 3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
