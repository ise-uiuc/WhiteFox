
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(31, 31)
x2 = torch.randn(31, 23)
inp = torch.randn(31, 22)
