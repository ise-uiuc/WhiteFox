
class m1(torch.nn.Module):
    def __init__(self, m2, p1):
        super().__init__()
        self.m2 = m2
        self.p1 = p1
    def forward(self, x1, x2):
        x3 = self.m2(x1) ** self.p1
        x4 = torch.nn.functional.dropout(x3)
        x5 = torch.rand_like(x4)
        x6 = self.m2(x2) ** self.p1
        return x5
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = x1 ** self.p1
        x3 = torch.randint(0, 9, (1,))
        x4 = torch.nn.functional.dropout(x2)
        return x4
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
