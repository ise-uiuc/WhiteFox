
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = F.dropout(x1, p=0.5, training=self.training)
        t2 = torch.rand_like(t1)
        return t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5, training=True)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
