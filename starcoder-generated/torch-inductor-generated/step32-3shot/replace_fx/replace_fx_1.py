
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        c1 = torch.rand_like(x)
        b1 = x * 2
        res = b1 * y
        c2 = torch.nn.functional.dropout(res, p=0.3)
        return c2
# Inputs to the model
x1 = torch.randn(1, 3, 4)
x2 = torch.randn(1, 3, 4)
