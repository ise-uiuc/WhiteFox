
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v1, w1, y, v2, t1, w2, w3, w4, w5):
        x1 = torch.nn.functional.dropout(y, p=0.7)
        x2 = torch.nn.functional.gelu(x1)
        z1 = torch.rand_like(y, dtype=torch.float)
        y1 = v1 + x2
        t2 = torch.nn.functional.dropout(t1, p=0.7)
        y2 = w1 + v2 + t2
        w6 = torch.rand_like(w1, dtype=torch.float)
        y3 = y1 + y2
        t3 = torch.nn.functional.silu(y3)
        u1 = torch.nn.functional.dropout(w5, p=0.7)
        z2 = t3 + w2 + w6 + z1
        t4 = torch.random.uniform(0., 1., size=(w1.size()[0],), dtype=torch.float)
        p1 = torch.rand_like(t4, dtype=torch.float)
        t5 = t4 + p1
        g1 = torch.nn.Hardsigmoid(z2)
        r1 = t5 + g1
        return r1
# Inputs to the model
v1 = torch.randn(1, 3, 2, 2)
w1 = torch.randn(1, 3, 2, 2)
y = torch.randn(1, 3, 2, 2)
v2 = torch.randn(1, 3, 2, 2)
t1 = torch.randn(1, 3, 2, 2)
w2 = torch.randn(1, 3, 2, 2)
w3 = torch.randn(1, 3, 2, 2)
w4 = torch.randn(1, 3, 2, 2)
w5 = torch.randn(1, 3, 2, 2)
