
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, weight):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        out = torch.matmul(v2, weight.t())
        return out
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(4, 6)
inp = torch.randn(2, 6)
weight = torch.randn(4, 2, 8)
