
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1 + x2 + x3, x3 + x1)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
x3 = torch.randn(10, 10)
inp = torch.randn(5, 10)
