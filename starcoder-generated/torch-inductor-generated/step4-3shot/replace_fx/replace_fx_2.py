
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.nn.functional.dropout(x, p=0.7)
        w1 = torch.rand_like(x, dtype=torch.float)
        y = v1 + y
        w2 = torch.rand_like(x, dtype=torch.float)
        t1 = y + w1
        z1 = torch.nn.functional.silu(t1)
        w5 = torch.rand_like(x, dtype=torch.float)
        t2 = v1 + w2
        w3 = torch.rand_like(x, dtype=torch.float)
        y1 = t2 + z1
        w4 = torch.rand_like(x, dtype=torch.float)
        z2 = y1 + w3
        return z2
# Inputs to the model
x = torch.randn(1, 3, 2, 2)
y = torch.randn(1, 3, 2, 2)
