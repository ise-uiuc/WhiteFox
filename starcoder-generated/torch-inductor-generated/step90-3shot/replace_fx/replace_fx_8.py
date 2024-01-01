
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
    def forward(self, x):
        x = torch.rand_like(x)
        return x
class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        module1_out = Module1().forward(x1)
        return x2 + module1_out
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module2 = Module2()
    def forward(self, x1):
        x2 = Module1().forward(x1)
        y1 = self.module2.forward(x2)
        y2 = self.module2.forward(y1)
        return y2 + x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
