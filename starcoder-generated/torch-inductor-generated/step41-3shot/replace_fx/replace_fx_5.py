
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1)
        x3 = F.dropout(x2, p=0.5)
        return x3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1)
        x3 = F.dropout(x2, p=0.5)
        x4 = F.dropout(x3, p=0.5)
        return x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1)
        x3 = F.dropout(x2)
        x4 = F.dropout(x3)
        x5 = F.dropout(x4, p=0.5)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
