
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp, x1, x2):
        v1 = inp * x1
        t = torch.mm(x2, inp)
        t = t * 1.2
        v2 = v1 + t
        return v2
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 3)
inp = torch.randn(3, 3)
