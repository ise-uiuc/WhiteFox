
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = F.dropout(x1, p=0.5, training=True)
        t2 = F.dropout(x2, p=0.5, training=True)
        return t1 + t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = F.dropout(x1, p=0.5, training=True)
        t2 = F.dropout(x2, p=0.8, training=True)
        return t1 + t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = F.dropout(x1, p=0.5, training=True)
        t2 = F.dropout(x2, p=0.8, training=False)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 1, 2)
