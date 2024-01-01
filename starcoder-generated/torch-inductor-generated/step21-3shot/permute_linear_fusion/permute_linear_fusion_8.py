
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        z = x.permute(1, 0, 2)
        z = self.linear(z)
        z = z.permute(1, 0, 2)
        return z

class C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.m = M()
    def forward(self, x):
       z = self.m(self.linear(x))
       return z

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.c = C()
    def forward(self, x):
       z = self.c(self.linear(x))
       return x / z
# Inputs to the model
x = torch.ones([1,1,1])
