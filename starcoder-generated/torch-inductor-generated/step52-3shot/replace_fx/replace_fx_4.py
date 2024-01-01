
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        b11 = F.dropout(x1, p=0.5)
        b12 = torch.rand_like(x1)
        b21 = F.dropout(x1, p=0.5)
        b22 = torch.rand_like(x1)
        z1 = (b11 * b12 + b21 * b22)
        return z1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)          # x2 not used at all
        x3 = F.dropout(x1, p=0.5)
        x4 = torch.rand_like(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
