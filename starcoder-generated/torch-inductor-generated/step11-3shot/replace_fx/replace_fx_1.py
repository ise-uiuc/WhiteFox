
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        c1 = torch.nn.functional.dropout(x, p=0.2)
        b1 = c1 * 2
        res = b1 + y
        c2 = torch.rand_like(b1)
        return res
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(1)
