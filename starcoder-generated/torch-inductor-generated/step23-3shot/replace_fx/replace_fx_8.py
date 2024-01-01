
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
    def forward(self, x1):
        x2 = x1 ** 2
        x3 = torch.randint(0, 9, (1,))
        with torch.no_grad():
            x4 = torch.rand_like(x3)
        x5 = self.m2(x4) # Invoke an nn.Module inside the input data flow scope
        return x5
class m2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 + torch.randint(0, 9, (1,))
        x3 = torch.rand_like(x1)
        x4 = torch.nn.functional.dropout(x2 + x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
