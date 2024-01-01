
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.5)
        y1 = torch.nn.functional.gelu(a1)
        w1 = torch.rand_like(x1, dtype=torch.float)
        # w2 = torch.rand_like(x1, dtype=torch.float)
        t1 = y1 + w1
        t2 = torch.nn.functional.silu(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
