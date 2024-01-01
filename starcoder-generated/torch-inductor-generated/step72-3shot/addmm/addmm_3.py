
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.add(torch.mm(x1, inp), x2)
        v2 = torch.mm(x3, inp)
        return v1, v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
inp = torch.randn(3, 3)
