
class Model(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x1):
        x2 = self.m(x1)
        x3 = torch.rand(size=(1,))
        x4 = torch.rand(size=(1,))
        x5 = x3 + x4
        return torch.rand_like(x5)
class m(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x1):
        x2 = self.a(x1)
        x3 = torch.rand(size=(1,))
        x4 = torch.rand(size=(1,))
        x5 = x3 + x4
        return torch.rand_like(x5)
class a(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randint((1,), 0, 2, dtype=torch.int32)
        return torch.nn.functional.relu(x2)
# Inputs to the model
x1 = torch.randn(1, 1, 2)
