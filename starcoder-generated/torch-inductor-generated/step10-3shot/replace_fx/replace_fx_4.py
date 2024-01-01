
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c):
        y = torch.nn.functional.dropout(b, p=0.5)
        x = torch.nn.functional.gelu(c)
        v1 = torch.rand_like(a, dtype=torch.float)
        w1 = torch.rand_like(b, dtype=torch.float)
        y = y + x
        x = torch.nn.functional.sigmoid(y)
        t = torch.nn.functional.silu(x)
        a = t + v1
        return a
# Inputs to the model
a = torch.randn(2, 2, 10)
b = torch.randn(2, 2, 10)
c = torch.randn(2, 2, 10)
