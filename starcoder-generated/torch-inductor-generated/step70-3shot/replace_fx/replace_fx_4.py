
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x2 = F.dropout(x2, p=0.5, training=True)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5, training=True)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x2)
        x4 = torch.rand_like(x2)
        x5 = x3 * x4
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
