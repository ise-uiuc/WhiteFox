
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1
        return v2
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(3, 3, 5)
inp = 1
