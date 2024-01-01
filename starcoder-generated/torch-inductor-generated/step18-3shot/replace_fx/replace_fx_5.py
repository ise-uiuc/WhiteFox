
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=0)
        t2 = torch.rand_like(x)
        t3 = torch.sum(t1 + t2)
        return t3
# Inputs to the model
x1 = torch.randn(1)
