
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        t1 = torch.rand_like(x1)
        t2 = torch.rand_like(t1)
        x4 = F.dropout(t2)
        return x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        t = torch.rand_like(x1)
        return t
# Inputs to the model
x1 = torch.randn(1, 2, 2)
