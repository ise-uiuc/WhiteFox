
class InputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(3, 3)
    def forward(self, x):
        return self.m1(x)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = InputModule()
    def forward(self, x1, x2):
        x1 = self.m(x1)
        x2 = self.m(x2)
        return torch.mm(x1, x2)
# Inputs to the model
m = InputModule()
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
