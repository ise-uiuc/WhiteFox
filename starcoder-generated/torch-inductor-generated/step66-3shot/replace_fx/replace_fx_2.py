
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 * x1
        t1 = torch.nn.functional.dropout(x2, p=0.4)
        t2 = torch.rand_like(t1)
        return t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 * x1
        x2 = torch.rand_like(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
