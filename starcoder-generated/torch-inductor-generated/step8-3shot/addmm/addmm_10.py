
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp, x2):
        v1 = torch.mm(x2, inp)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(1, 27)
x2 = torch.randn(35, 1)
inp = torch.randn(1, 35)
