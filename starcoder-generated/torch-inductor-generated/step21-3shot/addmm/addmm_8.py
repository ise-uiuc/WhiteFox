
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = inp + inp
        v2 = torch.mm(v2, v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 3)
inp = 1
