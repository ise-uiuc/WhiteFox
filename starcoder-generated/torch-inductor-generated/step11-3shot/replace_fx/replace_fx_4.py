
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.rand_like(x)
        a2 = torch.rand_like(x)
        c1 = torch.nn.functional.dropout(x)
        a3 = torch.mul(a1, a2)
        a4 = torch.rand_like(x)
        a = torch.mul(a3, a4)
        return a
# Inputs to the model
x1 = torch.randn(2)
