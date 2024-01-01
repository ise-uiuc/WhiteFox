
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp.T)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(222, 2)
x2 = torch.randn(2, 222)
inp = torch.randn(222, 2)
