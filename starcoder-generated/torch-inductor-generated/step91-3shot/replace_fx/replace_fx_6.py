
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = dropout(x, p=0.3)
        t2 = torch.nn.functional.gumbel_softmax(t1, tau=1.0)
        return t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = dropout(x, p=0.3)
        return t1 + 1.0
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = dropout(x1, p=0.3)
        x3 = dropout(x1, p=0.3) + 1.0
        x4 = dropout(x1, p=0.3) + 1.0
        x5 = dropout(x1, p=0.3) + 1.0
        x6 = dropout(x1, p=0.3) + 1.0
        x7 = dropout(x1, p=0.3) + 1.0
        x8 = dropout(x1, p=0.3) + 1.0
        x9 = dropout(x1, p=0.3) + 1.0
        x0 = dropout(x1, p=0.3) + 1.0
        x1 = dropout(x1, p=0.3) + 1.0
        x10 = torch.rand_like(x1)
        return x2 + x3 + x0 + x9 + x4 + x5
# Inputs to the model
x1 = torch.randn(2, 1, 2, 2)
