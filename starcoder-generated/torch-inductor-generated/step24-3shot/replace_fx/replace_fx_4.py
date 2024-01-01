
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.1)
        x3 = torch.nn.functional.dropout(x2, p=0.2)
        x4 = self.m2(x3, p1=1)
        return x4
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1, p1):
        x2 = x1 ** p1
        x3 = torch.nn.functional.dropout(x2, p=0.5)
        x4 = torch.nn.functional.dropout(x3, p=0.6)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
