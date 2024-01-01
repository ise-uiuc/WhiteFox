
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        z3 = torch.nn.functional.dropout(x1, p=0.5)
        y3 = torch.nn.functional.gelu(z3)
        w4 = torch.rand_like(x2, dtype=torch.float)
        t4 = y3 + w4
        return t4
# Inputs to the model
x1 = torch.randn(2, 2, 10)
x2 = torch.randn(2, 1, 10)
