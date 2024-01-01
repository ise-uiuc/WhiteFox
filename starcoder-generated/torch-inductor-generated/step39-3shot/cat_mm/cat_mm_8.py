
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m  = torch.nn.Linear(2, 2)
        self.l = [self.m]*10
    def forward(self, x1, x2):
        v = [torch.mm(self.m(x1*x2), self.m(x1*x2)) for loopVar1 in range(10)]
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(12, 6)
x2 = torch.randn(12, 6)
