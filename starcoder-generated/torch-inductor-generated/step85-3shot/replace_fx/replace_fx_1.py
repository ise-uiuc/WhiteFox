
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.randn(1, 2, 2)
        t2 = torch.randn(1, 2, 2)
        t3 = torch.randn(1, 2, 2)
        y = t1 * x * t3 * t2
        u = torch.rand_like(x)
        v = torch.nn.functional.dropout(u)
        w = torch.nn.functional.dropout(v)
        return v + w
# Inputs to the model
x = torch.randn(8, 3)
