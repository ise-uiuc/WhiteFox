
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.rand_like(x1)
        return x3
class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.rand_like(x1)
        x2 = torch.randint(4, (3,))
        x4 = torch.nn.functional.dropout(x3)
        return x4
class Model5(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.rand_like(x1)
        x5 = torch.rand_like(x1)
        x4 = torch.nn.functional.dropout(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
