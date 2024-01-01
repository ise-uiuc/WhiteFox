
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1, dtype=torch.float32)
        x4 = torch.nn.functional.dropout(x1, p=0.5)
        x5 = torch.rand_like(x4)
        return x5
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1, dtype=torch.float32)
        x4 = torch.nn.functional.dropout(x1, p=0.5)
        return x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1, dtype=torch.float32)
        return x3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
