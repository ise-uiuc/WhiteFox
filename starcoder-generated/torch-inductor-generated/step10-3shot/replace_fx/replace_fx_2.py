
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        w1 = torch.nn.functional.dropout(x, p=0.2)
        t2 = torch.rand_like(x, dtype=torch.float)
        y1 = w1 + t2
        x = y1 + x
        w3 = torch.nn.functional.dropout(x, p=0.9)
        y2 = torch.nn.functional.gelu(w3)
        w2 = torch.rand_like(x, dtype=torch.float)
        t3 = y2 + w2
        y = t3 + x
        w4 = torch.rand_like(x, dtype=torch.float)
        z1 = y + w4
        z2 = torch.nn.functional.silu(z1)
        return z2
# Inputs to the model
x = torch.randn(1, 3, 10)
