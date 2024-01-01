
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
    def forward(self, x1):
        x2 = self.m2(x1)
        x3 = torch.randint(0, 9, (1,))
        x4 = x3 + torch.rand_like(x3)
        return x4
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = x1 ** self.p1
        x3 = torch.randint(0, 9, (1,))
        a1 = torch.rand_like(x2)
        a2 = x2 - a1
        return torch.nn.functional.dropout(a2)
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
