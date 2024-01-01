
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
    def forward(self, x1):
        x2 = self.m2(x1)
        x3 = torch.addmm(torch.randn(1), x2, torch.mm(x2, x2))
        x4 = torch.nn.functional.dropout(x3)
        return x4
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = x2 - x1
        x4 = torch.randn(1) - x3
        return x4
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
