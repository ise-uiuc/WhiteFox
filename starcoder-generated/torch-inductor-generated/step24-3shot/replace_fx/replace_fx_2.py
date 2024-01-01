
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
        self.m2 = m2
    def forward(self, x1):
        x2 = self.m2(x1)
        x3 = torch.rand(1, dtype = torch.float32)
        x4 = x3**x3
        return x4
class m2(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
    def forward(self, x1):
        z = 2
        x2 = z * torch.nn.functional.dropout(x1, p = 0.1)
        x3 = torch.randn(1)
        x4 = x2 + x3
        x5 = x4**x3
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
