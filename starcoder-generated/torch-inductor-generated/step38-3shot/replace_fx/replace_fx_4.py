
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randint(5)
        t1 = torch.rand_like(x1)
        t1 = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 1.5], [2.1, -5.1, -8.2]]]])
        t2 = torch.rand_like(t1)
        x5 = F.dropout(t2)
        return x5
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randint(5)
        t1 = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 1.5], [2.1, -5.1, -8.2]]]])
        return t1
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
