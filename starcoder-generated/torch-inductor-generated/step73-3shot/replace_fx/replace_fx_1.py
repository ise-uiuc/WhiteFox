
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = torch.rand_like(x1)
        x4 = torch.rand_like(x1)
        return x3 + x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
        self.dropout2 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
