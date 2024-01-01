
import functools
class Model(torch.nn.Module):
    def __init__(self, loopVar):
        super().__init__()
        self.loopVar = loopVar
        functools.partial(self.partialForward)
    def forward(self, x1, x2):
        return self.partialForward(x1, x2)
    def partialForward(self, x1, x2):
        v = []
        for loopVar4 in range(self.loopVar):
            v.append(torch.mm(x1, x2))
        return torch.cat(v, 1)
loopVar = 1
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
