
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x, z):
        h = self.linear1(x)
        h.add_(self.linear2(x))
        h.sub_(self.linear1(z))
        h.add_(self.linear1(x))
        return h
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 5)
        self.linear2 = torch.nn.Linear(5, 2)
        self.custom = CustomModule()
    def forward(self, u, v, w):
        x = self.linear1(u)
        y = self.linear1(v)
        z = self.linear1(w)
        return self.custom(x, y)
# Inputs to the model
u = torch.randn(3, 2)
v = torch.randn(3, 2)
w = torch.randn(3, 2)
