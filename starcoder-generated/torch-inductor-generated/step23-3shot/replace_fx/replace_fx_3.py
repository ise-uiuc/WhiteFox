
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
        self.c1 = torch.nn.Conv2d(3, 4, 5)
    def forward(self, x1):
        x2 = torch.randint(0, 10, (1,))
        x3 = x1 ** x2
        x4 = torch.nn.functional.dropout(x3)
        x5 = torch.randint(0, 10, (1,))
        x6 = torch.rand_like(x4)
        x7 = self.c1(x6)
        x8 = torch.nn.functional.relu(x7)
        return x8
class m2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.rand(1)
        self.p2 = torch.nn.Parameter(torch.randn(1))
    def forward(self, x1):
        x2 = x1 ** self.p1
        x3 = torch.nn.functional.dropout(x2)
        x4 = torch.randint(0, 10, (1,))
        x5 = torch.rand_like(x3)
        x6 = self.p2 + x5
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 4)
