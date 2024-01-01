
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.rand_like(x1)
        self.m2 = torch.rand_like(x1)
    def forward(self, x1):
        x2 = F.dropout(x1)
        x3 = F.dropout(x2)
        x4 = F.dropout(x3)
        x5 = torch.rand_like(x2)
        x6 = F.dropout(x1)
        x7 = F.dropout(x6)
        x8 = F.dropout(x7)
        x9 = torch.rand_like(x2) + x2
        return x7
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1)
        x3 = torch.rand_like(x2) + x2
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
