
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.5)
        b = torch.nn.functional.dropout(x)
        c = a + b
        return c
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.5)
        a = a + x
        return a
# Inputs to the model
x1 = torch.randn(32, 32)
