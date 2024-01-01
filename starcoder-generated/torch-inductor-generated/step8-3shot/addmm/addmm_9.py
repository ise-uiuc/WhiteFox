
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, inp)
        v2 = torch.mm(x2, inp)
        return v2
# Inputs to the model
x1 = torch.randn(6, 4)
x2 = torch.randn(3, 3)
inp = torch.randn(4, 3)
