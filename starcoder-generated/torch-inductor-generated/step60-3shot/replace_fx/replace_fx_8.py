
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = F.dropout(x1, p=0.5)
        return t1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.rand_like(x1)
    def forward(self, x1):
        t2 = self.t1
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
