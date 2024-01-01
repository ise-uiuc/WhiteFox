
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = inp + x2
        v2 = torch.mm(x1, inp)
        v3 = (v1 + inp) * v2 + v2
        return v3
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5)
inp = torch.randn(5, 5, 5, 5)
