
class Model(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x, y):
        p = torch.rand_like(x)
        q = torch.rand_like(y)
        r = x * (y + p)
        return r
a = 1.2
# Inputs to the model
x = torch.randn(1)
y = torch.randn(2)
