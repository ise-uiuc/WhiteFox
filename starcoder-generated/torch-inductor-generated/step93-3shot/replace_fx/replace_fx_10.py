
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.tanh(x1)
        t1 = F.dropout(x2, p=0.5, training=self.training)
        t2 = torch.rand_like(t1)
        return t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.tanh(x1)
        x3 = x2 - 0.5
        t1 = torch.abs(x3)
        f1 = t1.float()
        t2 = F.dropout(f1, p=0.5, training=self.training)
        t3 = torch.rand_like(t2)
        return t3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x2 = torch.tanh(x1)
        x4 = F.dropout(x2, p=0.5, training=self.training)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
