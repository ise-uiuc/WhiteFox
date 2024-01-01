
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, arg):
        v1 = torch.mm(x1[::2], x2[1::2])
        v2 = v1[:2] + inp[2:-1:3, 1:-1:2]
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 3)
x2 = torch.randn(5, 5)
inp = torch.randn(5, 5)
arg = 0
