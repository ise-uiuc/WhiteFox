
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.pow(x, 2)
        dropout0 = F.dropout(t0)
        return dropout0
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.pow(x, 2)
        t1 = torch.rand_like(x)
        return (t0, t1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = F.dropout(x)
        t1 = torch.rand_like(x)
        return t0, t1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.rand_like(x)
        t1 = torch.pow(x, 2)
        return (t0, t1) 
# Inputs to the model
x = torch.randn((3,))
