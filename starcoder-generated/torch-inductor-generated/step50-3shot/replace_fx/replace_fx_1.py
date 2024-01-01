
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = F.dropout(x, p=0.5)
        b = F.dropout(a, p=0.25)
        c = F.dropout(b, p=0.75)
        return c
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = F.dropout(x, p=0.5)
        b = F.dropout(a, p=0.25)
        c = F.dropout(a, p=0.75)
        return c
# Inputs to the model
x1 = torch.randn(32, 32)
