
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        v1 = torch.mm(x, x.t())
        v2 = torch.mm(v1, x)
        v3 = v2 + v1
        v4 = v3.unsqueeze(1)
        v5 = torch.exp(v4)
        a1 = torch.add(v5, inp)
        a2, a3 = a1.shape
        return a1 + a2 + a3
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(2, 2, requires_grad=True)  # a tuple is passed as inp as a keyword argument
