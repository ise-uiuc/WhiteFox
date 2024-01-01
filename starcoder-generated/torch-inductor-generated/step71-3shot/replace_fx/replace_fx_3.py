
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.pow(x, 2)
        random = torch.rand_like(t0)
        t1 = torch.pow(random, 2)
        return t1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.pow(x, 2)
        t1 = torch.rand_like(t0)
        t2 = torch.pow(t1, 2)
        return t2
# Inputs to the model
x = torch.randn((10, 2, 2))
