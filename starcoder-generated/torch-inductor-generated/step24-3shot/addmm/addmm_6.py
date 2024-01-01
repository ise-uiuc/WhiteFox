
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = inp + inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 1000)
x2 = torch.randn(3, 3)
inp = torch.randn(1000, 3)
