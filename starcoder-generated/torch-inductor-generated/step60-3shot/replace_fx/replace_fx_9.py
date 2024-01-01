
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.rand_like(x1)
        b1 = F.dropout(x1, p=0.5)
        b2 = torch.rand_like(x1)
        t1 = F.dropout(x1, p=0.5)
        b3 = torch.rand_like(x1)
        b4 = torch.rand_like(x1)
        z1 = b1 * b2 + b3 + b4
        return z1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
