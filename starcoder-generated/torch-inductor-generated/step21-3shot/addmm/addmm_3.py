
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1
        v2 = v2 + inp
        v3 = v2 + inp
        return v3
# Inputs to the model
x1 = torch.randn(6,5)
x2 = torch.randn(5,3)
inp = 1
