
class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
    def forward(self, x):
        b1 = self.dropout(x)
        return b1
class ModelB(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b1 = x * 2
        c1 = torch.nn.functional.dropout(b1, p=0.5)
        c2 = torch.nn.functional.dropout(c1, p=0.5)
        c3 = c1 * 2
        return x
# Inputs to the model
x1 = torch.randn(1, 28)
# Model Ends

# Model Begins
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        p1 = torch.nn.functional.dropout(x, p=0.3)
        p2 = torch.nn.functional.dropout(p1, p=0.4)
        p3 = torch.nn.functional.dropout(y, p=0.3)
        p4 = torch.nn.functional.dropout(p2, p=0.7)
        p5 = torch.nn.functional.dropout(p3, p=0.2)
        p6 = torch.nn.functional.dropout(p4, p=0.2)
        return p5, p6
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 4)
# Model Ends