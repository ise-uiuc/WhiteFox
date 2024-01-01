
class m1(torch.nn.Module):
    def __init__(self, m2, v):
        super().__init__()
        self.m2 = m2
        self.v = v
    def forward(self, x1):
        x2 = self.m2(x1)
        x3 = torch.randint(0, 9, (1,))
        x4 = x2 ** self.v
        return x4
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = self.p1
        x3 = torch.randint(1, 9, (1,))
        x4 = x1 ** x3
        return x4
p1 = 1
v = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
